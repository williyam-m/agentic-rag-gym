"""
RAG Master Orchestrator - The core engine that drives multi-agent RAG workflows.

This orchestrator manages:
- Multi-agent coordination (retriever, reasoner, critic, planner, verifier)
- Episode lifecycle (reset, step, state)
- Process-aware reward computation
- Self-improvement through adversarial feedback loops
- Trajectory recording for RL training
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from rag_master.adapters import (
    BaseAgent,
    BaseDomainConfig,
    BaseLLMClient,
    BaseRewardFunction,
    BaseRetriever,
)
from rag_master.logging_config import get_logger
from rag_master.models import (
    AgentMessage,
    AgentRole,
    DifficultyLevel,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)
from rag_master.rewards import clamp_score

logger = get_logger(__name__)


class Orchestrator:
    """
    Central orchestrator for the Agentic RAG Gym.

    Manages the full lifecycle of an episode:
    1. reset() - Initialize a new episode with a task
    2. step(action) - Process an agent action and return observation/reward
    3. state() - Return current episode state

    Coordinates multiple agents, computes rewards with process supervision,
    and detects reward hacking attempts.
    """

    def __init__(
        self,
        domain_config: BaseDomainConfig,
        retriever: BaseRetriever,
        llm_client: BaseLLMClient,
        reward_function: BaseRewardFunction,
        agents: Optional[Dict[AgentRole, BaseAgent]] = None,
    ) -> None:
        self._domain = domain_config
        self._retriever = retriever
        self._llm = llm_client
        self._reward_fn = reward_function
        self._agents = agents or {}
        self._state: Optional[EpisodeState] = None
        self._trajectory: Optional[Trajectory] = None
        self._tasks = domain_config.get_tasks()

    @property
    def current_state(self) -> Optional[EpisodeState]:
        """Get current episode state."""
        return self._state

    @property
    def current_trajectory(self) -> Optional[Trajectory]:
        """Get current trajectory."""
        return self._trajectory

    @property
    def tasks(self) -> List[TaskDefinition]:
        """Get all available tasks."""
        return self._tasks

    async def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Specific task to load, or None for random selection.

        Returns:
            Initial observation dict.
        """
        task = self._select_task(task_id)
        episode_id = uuid4().hex[:16]

        self._state = EpisodeState(
            episode_id=episode_id,
            task=task,
            current_step=0,
            query_history=[],
            retrieved_docs=[],
            agent_messages=[],
            generated_answer="",
            intermediate_rewards=[],
            done=False,
            info={"task_difficulty": task.difficulty.value},
        )
        self._trajectory = Trajectory(
            episode_id=episode_id,
            task_id=task.task_id,
        )

        logger.info(
            "episode_reset",
            episode_id=episode_id,
            task_id=task.task_id,
            difficulty=task.difficulty.value,
        )

        return self._build_observation()

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an agent action and advance the environment.

        Args:
            action: Dict with 'type' and action-specific parameters.

        Returns:
            Dict with observation, reward, done, info.
        """
        if self._state is None or self._trajectory is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            return {
                "observation": self._build_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already completed."},
            }

        self._state.current_step += 1
        action_type = action.get("type", "unknown")

        # Create step record
        step = StepRecord(
            step_index=self._state.current_step,
            action_type=action_type,
            action_payload=action,
        )

        # Process action based on type
        if action_type == "retrieve":
            await self._handle_retrieve(action, step)
        elif action_type == "reason":
            await self._handle_reason(action, step)
        elif action_type == "answer":
            await self._handle_answer(action, step)
        elif action_type == "critique":
            await self._handle_critique(action, step)
        elif action_type == "plan":
            await self._handle_plan(action, step)
        elif action_type == "verify":
            await self._handle_verify(action, step)
        else:
            step.observation_summary = f"Unknown action type: {action_type}"
            step.intermediate_reward = 0.01

        # Compute step reward
        step_reward = await self._reward_fn.compute_step_reward(self._state, step)
        step.intermediate_reward = step_reward
        self._state.intermediate_rewards.append(step_reward)
        self._trajectory.steps.append(step)

        # Check termination
        done = self._check_done()
        self._state.done = done

        # Compute episode reward if done
        episode_reward = 0.0
        if done:
            episode_reward = await self._reward_fn.compute_episode_reward(
                self._trajectory, self._state
            )
            self._trajectory.total_reward = sum(
                s.intermediate_reward for s in self._trajectory.steps
            )
            self._trajectory.final_score = episode_reward
            self._trajectory.completed = True
            logger.info(
                "episode_completed",
                episode_id=self._state.episode_id,
                steps=self._state.current_step,
                score=episode_reward,
            )

        return {
            "observation": self._build_observation(),
            "reward": step_reward if not done else episode_reward,
            "done": done,
            "info": {
                "step": self._state.current_step,
                "action_type": action_type,
                "step_reward": step_reward,
                "episode_reward": episode_reward if done else None,
            },
        }

    def state(self) -> Dict[str, Any]:
        """Return the current environment state."""
        if self._state is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "episode_id": self._state.episode_id,
            "task_id": self._state.task.task_id,
            "task_name": self._state.task.name,
            "task_description": self._state.task.description,
            "difficulty": self._state.task.difficulty.value,
            "current_step": self._state.current_step,
            "max_steps": self._state.task.max_steps,
            "done": self._state.done,
            "query_history": self._state.query_history,
            "retrieved_doc_count": len(self._state.retrieved_docs),
            "answer_length": len(self._state.generated_answer),
            "intermediate_rewards": self._state.intermediate_rewards,
            "generated_answer": self._state.generated_answer,
        }

    async def grade(self, task_id: Optional[str] = None) -> float:
        """Grade the current episode using the domain grader."""
        if self._state is None or self._trajectory is None:
            return 0.01
        tid = task_id or self._state.task.task_id
        grader = self._domain.get_grader(tid)
        score = await grader.grade(self._state, self._trajectory)
        return clamp_score(score)

    # --- Private Methods ---

    def _select_task(self, task_id: Optional[str]) -> TaskDefinition:
        """Select a task by ID or return the first available."""
        if task_id:
            for task in self._tasks:
                if task.task_id == task_id:
                    return task
            raise ValueError(f"Task '{task_id}' not found. Available: {[t.task_id for t in self._tasks]}")
        if self._tasks:
            return self._tasks[0]
        raise ValueError("No tasks available in domain configuration.")

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation dict from current state."""
        if self._state is None:
            return {}
        return {
            "task": {
                "id": self._state.task.task_id,
                "name": self._state.task.name,
                "description": self._state.task.description,
                "difficulty": self._state.task.difficulty.value,
                "max_steps": self._state.task.max_steps,
            },
            "step": self._state.current_step,
            "retrieved_docs": [
                {
                    "content": r.document.content[:500],
                    "score": r.score,
                    "source": r.document.source,
                }
                for r in self._state.retrieved_docs[-5:]
            ],
            "query_history": self._state.query_history[-5:],
            "current_answer": self._state.generated_answer[:1000],
            "last_reward": (
                self._state.intermediate_rewards[-1]
                if self._state.intermediate_rewards
                else 0.0
            ),
            "done": self._state.done,
        }

    def _check_done(self) -> bool:
        """Check if the episode should terminate."""
        assert self._state is not None
        if self._state.current_step >= self._state.task.max_steps:
            return True
        if self._state.generated_answer and self._state.current_step >= 3:
            last_action = (
                self._trajectory.steps[-1].action_type
                if self._trajectory and self._trajectory.steps
                else ""
            )
            if last_action == "answer":
                return True
        return False

    async def _handle_retrieve(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle a retrieval action."""
        assert self._state is not None
        query = action.get("query", self._state.task.description)
        self._state.query_history.append(query)

        if AgentRole.RETRIEVER in self._agents:
            result, messages = await self._agents[AgentRole.RETRIEVER].act(
                self._state,
                [AgentMessage(sender=AgentRole.PLANNER, content=query, message_type="query")],
            )
            self._state.retrieved_docs = result.get("results", [])
            self._state.agent_messages.extend(messages)
        else:
            results = await self._retriever.retrieve(query, top_k=5)
            self._state.retrieved_docs = results

        step.observation_summary = f"Retrieved {len(self._state.retrieved_docs)} documents"
        step.reasoning_trace = f"Query: {query}"

    async def _handle_reason(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle a reasoning action."""
        assert self._state is not None
        if AgentRole.REASONER in self._agents:
            result, messages = await self._agents[AgentRole.REASONER].act(
                self._state, self._state.agent_messages[-5:]
            )
            reasoning = result.get("reasoning", "")
            self._state.agent_messages.extend(messages)
        else:
            context = self._build_reasoning_context()
            reasoning = await self._llm.generate(
                [
                    {"role": "system", "content": self._domain.get_system_prompt()},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
            )
        step.reasoning_trace = reasoning
        step.observation_summary = f"Reasoning completed ({len(reasoning)} chars)"

    async def _handle_answer(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle an answer submission action."""
        assert self._state is not None
        answer = action.get("answer", "")
        if not answer:
            context = self._build_answer_context()
            answer = await self._llm.generate(
                [
                    {"role": "system", "content": self._domain.get_system_prompt()},
                    {"role": "user", "content": context},
                ],
                temperature=0.2,
            )
        self._state.generated_answer = answer
        step.observation_summary = f"Answer submitted ({len(answer)} chars)"
        step.reasoning_trace = answer[:300]

    async def _handle_critique(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle a critique action."""
        assert self._state is not None
        if AgentRole.CRITIC in self._agents:
            result, messages = await self._agents[AgentRole.CRITIC].act(
                self._state, self._state.agent_messages[-5:]
            )
            critique = result.get("critique", "")
            self._state.agent_messages.extend(messages)
        else:
            critique = "No critic agent configured."
        step.reasoning_trace = critique
        step.observation_summary = "Critique completed"

    async def _handle_plan(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle a planning action."""
        assert self._state is not None
        if AgentRole.PLANNER in self._agents:
            result, messages = await self._agents[AgentRole.PLANNER].act(
                self._state, []
            )
            plan = result.get("plan", "")
            self._state.agent_messages.extend(messages)
        else:
            plan = "Default plan: retrieve → reason → answer"
        step.reasoning_trace = plan
        step.observation_summary = "Plan created"

    async def _handle_verify(self, action: Dict[str, Any], step: StepRecord) -> None:
        """Handle a verification action."""
        assert self._state is not None
        if AgentRole.VERIFIER in self._agents:
            result, messages = await self._agents[AgentRole.VERIFIER].act(
                self._state, []
            )
            verification = result.get("verification", "")
        else:
            verification = "No verifier agent configured."
        step.reasoning_trace = verification
        step.observation_summary = "Verification completed"

    def _build_reasoning_context(self) -> str:
        """Build reasoning context from retrieved docs."""
        assert self._state is not None
        parts = [f"Task: {self._state.task.description}"]
        if self._state.retrieved_docs:
            parts.append("Retrieved Documents:")
            for r in self._state.retrieved_docs[:5]:
                parts.append(f"- [{r.score:.2f}] {r.document.content[:400]}")
        parts.append("Analyze the documents and synthesize a comprehensive response.")
        return "\n\n".join(parts)

    def _build_answer_context(self) -> str:
        """Build context for generating a final answer."""
        assert self._state is not None
        parts = [f"Task: {self._state.task.description}"]
        if self._state.retrieved_docs:
            parts.append("Key Information:")
            for r in self._state.retrieved_docs[:3]:
                parts.append(f"- {r.document.content[:300]}")
        reasoning_traces = [
            s.reasoning_trace
            for s in (self._trajectory.steps if self._trajectory else [])
            if s.reasoning_trace and s.action_type == "reason"
        ]
        if reasoning_traces:
            parts.append(f"Analysis:\n{reasoning_traces[-1][:500]}")
        parts.append("Provide a complete, well-structured answer based on the above.")
        return "\n\n".join(parts)
