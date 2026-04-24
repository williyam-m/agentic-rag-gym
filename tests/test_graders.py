"""Tests for the aerospace domain graders."""

from __future__ import annotations

import pytest

from domains.aerospace.graders import (
    DebrisMitigationGrader,
    HypersonicVehicleGrader,
    LifeSupportGrader,
    MarsEDLGrader,
    PropulsionComparisonGrader,
    get_grader,
)
from rag_master.models import (
    DifficultyLevel,
    Document,
    EpisodeState,
    RetrievalResult,
    StepRecord,
    TaskDefinition,
    Trajectory,
)
from rag_master.rewards import _SCORE_MAX, _SCORE_MIN


def _make_task(task_id: str = "test", difficulty: DifficultyLevel = DifficultyLevel.EASY) -> TaskDefinition:
    return TaskDefinition(
        task_id=task_id, name="Test", description="Test task",
        difficulty=difficulty, max_steps=10,
    )


def _make_state(answer: str, task_id: str = "test") -> EpisodeState:
    return EpisodeState(task=_make_task(task_id), generated_answer=answer)


def _make_trajectory(steps: int = 3) -> Trajectory:
    return Trajectory(
        task_id="test",
        steps=[
            StepRecord(step_index=i, action_type=["retrieve", "reason", "answer"][i % 3])
            for i in range(steps)
        ],
    )


class TestPropulsionComparisonGrader:
    """Tests for propulsion comparison grading."""

    @pytest.fixture
    def grader(self) -> PropulsionComparisonGrader:
        return PropulsionComparisonGrader()

    @pytest.mark.asyncio
    async def test_empty_answer_min_score(self, grader: PropulsionComparisonGrader) -> None:
        state = _make_state("")
        score = await grader.grade(state, _make_trajectory())
        assert score == _SCORE_MIN

    @pytest.mark.asyncio
    async def test_good_answer_high_score(self, grader: PropulsionComparisonGrader) -> None:
        answer = (
            "Ion propulsion achieves a specific impulse of 4,190 seconds using the NEXT xenon thruster, "
            "compared to nuclear thermal propulsion which achieves 850-1,000 seconds. However, ion engines "
            "produce very low thrust of 0.5 newton, making them unsuitable for crewed missions. "
            "Nuclear thermal propulsion, demonstrated by NASA's DRACO program with DARPA, uses HALEU "
            "hydrogen reactor to reduce Mars transit duration from 9 months to 4 months. "
            "I recommend nuclear thermal for crewed Mars missions due to higher thrust and technology readiness."
        )
        state = _make_state(answer)
        score = await grader.grade(state, _make_trajectory())
        assert score > 0.35

    @pytest.mark.asyncio
    async def test_partial_answer_medium_score(self, grader: PropulsionComparisonGrader) -> None:
        answer = "Ion propulsion has high specific impulse of 4,190 seconds. Nuclear thermal propulsion from NASA DRACO."
        state = _make_state(answer)
        score = await grader.grade(state, _make_trajectory())
        assert _SCORE_MIN < score < _SCORE_MAX

    @pytest.mark.asyncio
    async def test_score_bounds(self, grader: PropulsionComparisonGrader) -> None:
        for answer in ["short", "x" * 10, "random text that has no keywords"]:
            state = _make_state(answer)
            score = await grader.grade(state, _make_trajectory())
            assert _SCORE_MIN <= score <= _SCORE_MAX


class TestDebrisMitigationGrader:
    """Tests for debris mitigation grading."""

    @pytest.fixture
    def grader(self) -> DebrisMitigationGrader:
        return DebrisMitigationGrader()

    @pytest.mark.asyncio
    async def test_comprehensive_answer(self, grader: DebrisMitigationGrader) -> None:
        answer = (
            "The Kessler syndrome describes a cascade of collisions increasing debris density. "
            "ESA tracks over 36,500 objects in LEO. Active debris removal technologies include "
            "electrodynamic tethers, laser ablation, and robotic capture like ClearSpace-1. "
            "The economic cost is estimated at $500 million to $1 billion per year. "
            "I recommend prioritizing robotic capture as the most effective approach to implement."
        )
        state = _make_state(answer)
        score = await grader.grade(state, _make_trajectory())
        assert score > 0.4


class TestMarsEDLGrader:
    """Tests for Mars EDL grading."""

    @pytest.fixture
    def grader(self) -> MarsEDLGrader:
        return MarsEDLGrader()

    @pytest.mark.asyncio
    async def test_comprehensive_answer(self, grader: MarsEDLGrader) -> None:
        answer = (
            "The Mars atmosphere presents a paradox: thick enough for heating but too thin for deceleration. "
            "The EDL sequence involves aerocapture, aerobraking, entry, descent, and landing phases. "
            "Thermal protection using PICA heat shield handles heat flux up to 200 W/cm². "
            "Supersonic retropropulsion demonstrated by Falcon 9 enables heavy payload landing. "
            "Terrain-relative navigation with LIDAR from Perseverance provides landing accuracy. "
            "This integrated design combines multiple systems into a coherent architecture."
        )
        state = _make_state(answer)
        score = await grader.grade(state, _make_trajectory())
        assert score > 0.4


class TestGetGrader:
    """Tests for grader registry."""

    def test_valid_task_ids(self) -> None:
        valid_ids = [
            "aero_easy_propulsion_comparison",
            "aero_easy_debris_mitigation",
            "aero_medium_mars_edl",
            "aero_medium_life_support",
            "aero_hard_hypersonic_vehicle",
        ]
        for task_id in valid_ids:
            grader = get_grader(task_id)
            assert grader is not None

    def test_invalid_task_id(self) -> None:
        with pytest.raises(ValueError, match="No grader"):
            get_grader("nonexistent_task")
