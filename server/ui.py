"""
Gradio UI for Agentic RAG Gym - Royal Glassmorphism Theme.

Provides an interactive interface for:
- Running episodes against the environment
- Viewing task descriptions and difficulty
- Real-time step-by-step feedback with rewards
- Trajectory visualization
- Benchmark results
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr
import httpx

ROYAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --gold: #D4AF37;
    --gold-light: #F0D060;
    --gold-dark: #A08520;
    --royal-black: #0A0A0F;
    --royal-dark: #12121A;
    --royal-surface: #1A1A2E;
    --royal-card: rgba(26, 26, 46, 0.7);
    --text-primary: #FFFFFF;
    --text-secondary: #B0B0C0;
    --glass-border: rgba(212, 175, 55, 0.3);
    --glass-bg: rgba(26, 26, 46, 0.6);
    --success: #4CAF50;
    --warning: #FF9800;
    --error: #F44336;
}

.gradio-container {
    background: linear-gradient(135deg, var(--royal-black) 0%, var(--royal-dark) 50%, #16213E 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(180deg, rgba(212, 175, 55, 0.1) 0%, transparent 100%);
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 1.5rem;
}

.main-header h1 {
    font-family: 'Cinzel', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--gold-light), var(--gold), var(--gold-dark));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: 2px;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 300;
}

.glass-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.gold-button {
    background: linear-gradient(135deg, var(--gold-dark), var(--gold), var(--gold-light)) !important;
    color: var(--royal-black) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
}

.gold-button:hover {
    box-shadow: 0 6px 25px rgba(212, 175, 55, 0.5) !important;
    transform: translateY(-2px) !important;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.badge-easy { background: rgba(76, 175, 80, 0.2); color: #4CAF50; border: 1px solid rgba(76, 175, 80, 0.4); }
.badge-medium { background: rgba(255, 152, 0, 0.2); color: #FF9800; border: 1px solid rgba(255, 152, 0, 0.4); }
.badge-hard { background: rgba(244, 67, 54, 0.2); color: #F44336; border: 1px solid rgba(244, 67, 54, 0.4); }

.reward-display {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    text-align: center;
    padding: 1rem;
}

.step-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    padding: 1rem;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    border: 1px solid rgba(212, 175, 55, 0.15);
    max-height: 400px;
    overflow-y: auto;
}

.metric-card {
    text-align: center;
    padding: 1rem;
    background: rgba(212, 175, 55, 0.05);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 12px;
}

.metric-card .value {
    font-family: 'Cinzel', serif;
    font-size: 1.8rem;
    color: var(--gold);
}

.metric-card .label {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.3rem;
}

/* Override Gradio defaults */
.dark .gr-box, .dark .gr-input, .dark .gr-panel {
    background: var(--glass-bg) !important;
    border-color: var(--glass-border) !important;
}

.dark label, .dark .gr-check-radio label {
    color: var(--text-secondary) !important;
}

textarea, input[type="text"], select {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.gr-button.primary {
    background: linear-gradient(135deg, var(--gold-dark), var(--gold)) !important;
    color: var(--royal-black) !important;
    border: none !important;
}
"""

API_URL = "http://localhost:7860"

# --- Task choices per domain ---
DOMAIN_TASKS = {
    "aerospace": [
        ("Compare Propulsion Technologies (Easy)", "aero_easy_propulsion_comparison"),
        ("Space Debris Mitigation (Easy)", "aero_easy_debris_mitigation"),
        ("Mars EDL Architecture (Medium)", "aero_medium_mars_edl"),
        ("Deep Space Life Support (Medium)", "aero_medium_life_support"),
        ("Hypersonic Vehicle Design (Hard)", "aero_hard_hypersonic_vehicle"),
    ],
    "legal_research": [
        ("Contract Clause Analysis (Easy)", "legal_easy_contract_review"),
        ("Data Privacy Compliance (Easy)", "legal_easy_privacy_compliance"),
        ("IP Assessment (Medium)", "legal_medium_ip_analysis"),
        ("M&A Due Diligence (Medium)", "legal_medium_ma_due_diligence"),
        ("Cross-Border Dispute (Hard)", "legal_hard_cross_border_dispute"),
    ],
}

DOMAIN_LABELS = {
    "aerospace": "Aerospace Research",
    "legal_research": "Legal Research",
}


def _call_api(method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make an API call to the backend."""
    url = f"{API_URL}{endpoint}"
    try:
        with httpx.Client(timeout=60.0) as client:
            if method == "GET":
                resp = client.get(url)
            else:
                resp = client.post(url, json=data or {})
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def switch_domain(domain: str):
    """Switch the active domain and return updated task choices for both dropdowns."""
    result = _call_api("POST", "/domain/switch", {"domain": domain})
    if "error" in result:
        fallback = DOMAIN_TASKS.get("aerospace", [])
        return (
            gr.update(choices=fallback, value=None),
            gr.update(choices=fallback, value=None),
            f"Error: {result['error']}",
        )
    tasks = DOMAIN_TASKS.get(domain, [])
    default_val = tasks[0][1] if tasks else None
    label = DOMAIN_LABELS.get(domain, domain)
    return (
        gr.update(choices=tasks, value=default_val),
        gr.update(choices=tasks, value=default_val),
        f"Switched to {label} domain",
    )


def reset_environment(task_id: str) -> tuple:
    """Reset the environment with the selected task."""
    data = {"task_id": task_id} if task_id else {}
    result = _call_api("POST", "/reset", data)

    if "error" in result:
        return (
            f"**Error:** {result['error']}",
            "N/A",
            "",
            "0",
            "0.0",
            "Not started",
        )

    obs = result.get("observation", {})
    task = obs.get("task", {})

    task_info = (
        f"### {task.get('name', 'Unknown')}\n\n"
        f"**Difficulty:** {task.get('difficulty', 'N/A')}\n\n"
        f"**Max Steps:** {task.get('max_steps', 'N/A')}\n\n"
        f"**Description:**\n{task.get('description', 'N/A')}"
    )

    difficulty = task.get("difficulty", "unknown")
    badge = f'<span class="status-badge badge-{difficulty}">{difficulty.upper()}</span>'

    return (
        task_info,
        badge,
        "",
        "0",
        "0.0",
        "Episode started - take your first action!",
    )


def take_step(action_type: str, query: str, answer: str) -> tuple:
    """Take a step in the environment."""
    action_data: Dict[str, Any] = {"type": action_type}
    if query:
        action_data["query"] = query
    if answer:
        action_data["answer"] = answer

    result = _call_api("POST", "/step", action_data)

    if "error" in result:
        return ("", "0", "0.0", f"**Error:** {result['error']}", "")

    obs = result.get("observation", {})
    reward = result.get("reward", 0.0)
    done = result.get("done", False)
    info = result.get("info", {})

    # Build step log
    step_num = info.get("step", 0)
    step_log = (
        f"**Step {step_num}** | Action: `{action_type}` | "
        f"Reward: `{reward:.4f}` | Done: `{done}`"
    )

    # Retrieved docs display
    docs_display = ""
    if obs.get("retrieved_docs"):
        docs_display = "\n\n".join(
            f"**[{d['score']:.2f}]** {d['content'][:200]}..."
            for d in obs["retrieved_docs"]
        )

    current_answer = obs.get("current_answer", "")
    status = "Episode Complete!" if done else f"Step {step_num} completed"

    return (
        docs_display,
        str(step_num),
        f"{reward:.4f}",
        status,
        current_answer,
    )


def grade_episode() -> str:
    """Grade the current episode."""
    result = _call_api("POST", "/grade")
    if "error" in result:
        return f"**Error:** {result['error']}"
    score = result.get("score", 0.0)
    task_id = result.get("task_id", "Unknown")
    return (
        f'<div class="reward-display" style="color: var(--gold);">'
        f"Score: {score:.4f}"
        f"</div>\n\n"
        f"**Task:** {task_id}\n\n"
        f"**Episode:** {result.get('episode_id', 'N/A')}"
    )


def get_tasks() -> str:
    """Get list of available tasks."""
    result = _call_api("GET", "/tasks")
    if "error" in result:
        return f"Error: {result['error']}"
    tasks = result.get("tasks", [])
    lines = []
    for t in tasks:
        difficulty = t.get("difficulty", "unknown")
        lines.append(
            f"| `{t['task_id']}` | {t['name']} | "
            f'<span class="status-badge badge-{difficulty}">{difficulty}</span> | '
            f"{t['max_steps']} |"
        )
    header = "| Task ID | Name | Difficulty | Max Steps |\n|---|---|---|---|\n"
    return header + "\n".join(lines)


def run_full_episode(task_id: str) -> tuple:
    """Run a complete automated episode."""
    reset_result = _call_api("POST", "/reset", {"task_id": task_id} if task_id else {})
    if "error" in reset_result:
        return (f"Error: {reset_result['error']}", "", "")

    log_lines = ["**Episode Started**\n"]
    steps = [
        {"type": "plan"},
        {"type": "retrieve", "query": reset_result.get("observation", {}).get("task", {}).get("description", "")},
        {"type": "reason"},
        {"type": "critique"},
        {"type": "retrieve", "query": "detailed technical analysis"},
        {"type": "reason"},
        {"type": "answer"},
    ]

    total_reward = 0.0
    for i, action in enumerate(steps, 1):
        result = _call_api("POST", "/step", action)
        if "error" in result:
            log_lines.append(f"Step {i}: Error - {result['error']}")
            break
        reward = result.get("reward", 0.0)
        total_reward += reward
        done = result.get("done", False)
        # Mark done on the final answer step since orchestrator only flags
        # done on the step *after* answer, which auto pilot never takes.
        if action["type"] == "answer" and i == len(steps):
            done = True
        log_lines.append(
            f"**Step {i}** | `{action['type']}` | Reward: `{reward:.4f}` | Done: `{done}`"
        )
        if done:
            break

    # Grade
    grade_result = _call_api("POST", "/grade")
    final_score = grade_result.get("score", 0.0) if "error" not in grade_result else 0.0

    log_text = "\n\n".join(log_lines)
    state = _call_api("GET", "/state")
    answer = state.get("generated_answer", "")

    score_display = (
        f'<div class="reward-display" style="color: var(--gold);">'
        f"Final Score: {final_score:.4f}"
        f"</div>"
    )

    return (log_text, answer, score_display)


def build_ui() -> gr.Blocks:
    """Build the Gradio UI with royal glassmorphism theme."""
    with gr.Blocks(css=ROYAL_CSS, theme=gr.themes.Base(), title="Agentic RAG Gym") as demo:
        # Header
        gr.HTML(
            '<div class="main-header">'
            "<h1>⚜ AGENTIC RAG GYM ⚜</h1>"
            "<p>RL-Enhanced Agentic RAG Framework — Extensible to Any Domain</p>"
            "</div>"
        )

        # Domain selector - top level
        with gr.Row():
            domain_selector = gr.Dropdown(
                choices=[
                    ("Aerospace Research", "aerospace"),
                    ("Legal Research", "legal_research"),
                ],
                label="Select Domain",
                value="aerospace",
                scale=2,
            )
            domain_status = gr.Textbox(label="Domain Status", value="Active: Aerospace Research", interactive=False, scale=2)
            gr.HTML(
                '<div style="padding: 10px; color: var(--text-secondary); font-size: 0.85rem; text-align: center;">'
                '🔮 More domains coming soon — the RAG Master framework is designed for any domain'
                '</div>'
            )

        with gr.Tabs() as tabs:
            # --- Tab 1: Interactive Mode ---
            with gr.Tab("🎯 Interactive Mode", id="interactive"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="glass-card"><h3 style="color: var(--gold);">Environment Control</h3></div>')
                        task_dropdown = gr.Dropdown(
                            choices=DOMAIN_TASKS["aerospace"],
                            label="Select Task",
                            value="aero_easy_propulsion_comparison",
                        )
                        reset_btn = gr.Button("⚡ Reset Episode", variant="primary", elem_classes=["gold-button"])
                        difficulty_badge = gr.HTML(value="")

                        gr.HTML('<hr style="border-color: rgba(212,175,55,0.2); margin: 1rem 0;">')

                        action_type = gr.Radio(
                            choices=["plan", "retrieve", "reason", "critique", "verify", "answer"],
                            label="Action Type",
                            value="retrieve",
                        )
                        query_input = gr.Textbox(
                            label="Query (for retrieve actions)",
                            placeholder="Enter search query...",
                            lines=2,
                        )
                        answer_input = gr.Textbox(
                            label="Answer (for answer actions)",
                            placeholder="Enter your answer...",
                            lines=4,
                        )
                        step_btn = gr.Button("▶ Take Step", variant="primary", elem_classes=["gold-button"])
                        grade_btn = gr.Button("📊 Grade Episode", variant="secondary")

                    with gr.Column(scale=2):
                        task_info = gr.Markdown(label="Task Information", value="*Select a task and click Reset*")
                        with gr.Row():
                            step_counter = gr.Textbox(label="Step", value="0", interactive=False)
                            reward_display = gr.Textbox(label="Last Reward", value="0.0", interactive=False)
                        status_display = gr.Textbox(label="Status", value="Ready", interactive=False)
                        docs_display = gr.Markdown(label="Retrieved Documents", value="")
                        answer_display = gr.Textbox(label="Generated Answer", value="", lines=6, interactive=False)
                        grade_display = gr.HTML(label="Grade", value="")

                reset_btn.click(
                    fn=reset_environment,
                    inputs=[task_dropdown],
                    outputs=[task_info, difficulty_badge, docs_display, step_counter, reward_display, status_display],
                )
                step_btn.click(
                    fn=take_step,
                    inputs=[action_type, query_input, answer_input],
                    outputs=[docs_display, step_counter, reward_display, status_display, answer_display],
                )
                grade_btn.click(fn=grade_episode, outputs=[grade_display])

            # --- Tab 2: Auto Pilot ---
            with gr.Tab("🚀 Auto Pilot", id="autopilot"):
                gr.HTML(
                    '<div class="glass-card"><h3 style="color: var(--gold);">Automated Episode Runner</h3>'
                    "<p>Runs a complete episode automatically: Plan → Retrieve → Reason → Critique → Retrieve → Reason → Answer</p></div>"
                )
                with gr.Row():
                    auto_task = gr.Dropdown(
                        choices=DOMAIN_TASKS["aerospace"],
                        label="Select Task",
                        value="aero_easy_propulsion_comparison",
                    )
                    auto_run_btn = gr.Button("🏁 Run Full Episode", variant="primary", elem_classes=["gold-button"])

                auto_log = gr.Markdown(label="Episode Log", value="")
                auto_answer = gr.Textbox(label="Final Answer", value="", lines=8, interactive=False)
                auto_score = gr.HTML(label="Final Score", value="")

                auto_run_btn.click(
                    fn=run_full_episode,
                    inputs=[auto_task],
                    outputs=[auto_log, auto_answer, auto_score],
                )

            # --- Tab 3: Tasks ---
            with gr.Tab("📋 Tasks", id="tasks"):
                gr.HTML(
                    '<div class="glass-card"><h3 style="color: var(--gold);">Available Tasks</h3>'
                    "<p>Tasks for the currently active domain — switch domains above to see different tasks</p></div>"
                )
                tasks_display = gr.Markdown(value="Click refresh to load tasks")
                refresh_btn = gr.Button("🔄 Refresh Tasks", variant="secondary")
                refresh_btn.click(fn=get_tasks, outputs=[tasks_display])

            # --- Tab 4: About ---
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown(
                    """
                    <div class="glass-card">

                    ## ⚜ Agentic RAG Gym

                    An **open-source RL-enhanced framework** that revolutionizes
                    **Retrieval-Augmented Generation** by training agents through
                    reinforcement learning across any knowledge domain.

                    ### Architecture

                    | Component | Technology |
                    |---|---|
                    | Framework | RAG Master (custom orchestrator) |
                    | Backend | FastAPI + Uvicorn |
                    | Vector Store | FAISS |
                    | Embeddings | sentence-transformers |
                    | LLM | OpenAI-compatible API |
                    | UI | Gradio (HF Space) |
                    | Domains | Aerospace Research, Legal Research, _more coming soon_ |

                    ### Multi-Agent System

                    - **Retriever Agent** — Document search and query reformulation
                    - **Reasoner Agent** — Analysis and synthesis over documents
                    - **Critic Agent** — Quality evaluation and improvement suggestions
                    - **Planner Agent** — Strategic planning for complex tasks
                    - **Verifier Agent** — Factual grounding verification

                    ### OpenEnv Compliance

                    Full implementation of the OpenEnv specification:
                    `reset()` → `step()` → `state()` → `grade()`

                    </div>
                    """
                )

        # Wire domain selector to update both task dropdowns in one call
        domain_selector.change(
            fn=switch_domain,
            inputs=[domain_selector],
            outputs=[task_dropdown, auto_task, domain_status],
        )

    return demo
