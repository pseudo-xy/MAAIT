"""USV decision agents ("policies").

An AGENT-mode USV is driven by a *policy* — the "brain" that makes the choices:
which boat to go after, what speed and direction, when to identify, and whether
to circle or escort after identification. The *mechanics* of carrying those out
(how to travel to a point, how to orbit) live in usv_movement.py and are shared
by every policy.

This is the seam where trained agents plug in. SCRIPTED is the rule-based
mechanics layer. MDP is the solved engagement MDP (reschu.mdp). LLM is still a
stub that falls back to the scripted behavior and is clearly marked as such.

Each USV chooses its policy via `usv.mode` (USVMode.LLM / USVMode.MDP); MANUAL
USVs are driven by the operator, not a policy.

To add a real agent later:
  * implement `decide()` in LLMPolicy / MDPPolicy (or a new Policy subclass),
  * have it return the USV's next position for this tick (using the helpers in
    usv_movement.py for the actual motion),
  * everything else — the engine loop, identification, logging — stays the same.
"""

from __future__ import annotations

from .models import USV, World
from .observation import Observation, observe
from .types import USVMode, Vec2
from .usv_movement import assigned_agent_step


class Policy:
    """Base class for a USV decision agent.

    A policy decides from a truth-free `Observation` (what the USV perceives) and
    returns where the USV should be after this tick. It may set fields on the USV
    that the rest of the system reads (assigned_target_id, operator_speed_knots,
    circling_since_min, pending_declaration) — those represent the agent's
    choices. `pending_declaration = (boat_id, Perception.X)` is how an agent
    commits to a class; the engine applies it only while the boat is in ID range.

    Ground-truth hardening: real agents override `act(usv, obs, dt_min)`, which is
    handed ONLY the truth-free observation — never the World. An agent therefore
    *cannot* reach a boat's true class, behavior, or true threat tier even by
    accident: the world object is not in scope inside `act()`. This upgrades the
    guarantee from "agents don't read truth" to "agents can't read truth", which
    matters once real LLM/MDP agents are written.

    (The internal scripted baseline is the one exception — see ScriptedPolicy.)
    """

    name: str = "policy"

    def decide(self, usv: USV, world: World, dt_min: float, now_min: float) -> Vec2:
        """Build the truth-free observation, then act on it. Real agents override
        `act()` (which never sees `world`), NOT this method."""
        obs = observe(usv, world, now_min)
        return self.act(usv, obs, dt_min)

    def act(self, usv: USV, obs: Observation, dt_min: float) -> Vec2:
        """Decide the USV's next position from the truth-free observation ONLY.

        Note there is no `world` parameter by design — an agent's decision must
        not depend on ground truth, and this signature makes that impossible
        rather than merely discouraged. Override this in real LLM/MDP agents.
        """
        raise NotImplementedError


class ScriptedPolicy(Policy):
    """Internal rule-based baseline. NOT a selectable mode — it's the shared
    logic the untrained LLM/MDP stubs currently run on so the sim works.

    Behavior: go to the assigned boat -> once identified, circle a perceived
    enemy for a fixed time or escort a perceived friendly/civilian off the grid
    -> release and let the engine re-assign. When the real LLM/MDP agents are
    implemented, they replace this fallback with their own decisions.

    Exception to the world-hiding rule: as an INTERNAL mechanics component (not a
    competing agent under study), the baseline overrides `decide()` directly so it
    can use the shared movement helpers, which take `world` to carry out motion.
    It still reads no ground truth — assigned_agent_step and usv_movement use
    perceived class only (verified: no true_class/behavior access anywhere in the
    movement layer). `world` is used here for MECHANICS (moving, circling), never
    to see a boat's true identity.
    """

    name = "scripted"

    def decide(self, usv: USV, world: World, dt_min: float, now_min: float) -> Vec2:
        return assigned_agent_step(usv, world, dt_min, now_min)


class LLMPolicy(Policy):
    """LLM-driven agent — NOT TRAINED / CONNECTED YET (stub).

    When implemented, override `act(usv, obs, dt_min)`: build a description of the
    USV's view from `obs` (nearby boats, their PERCEIVED classes, distances, etc.
    — obs has no ground truth), send it to an LLM, parse the LLM's chosen action
    (target boat, speed, next move, identify / circle / escort), and return the
    next position. Because `act()` never receives `world`, the LLM cannot be fed a
    boat's true identity even by mistake.

    For now it falls back to the scripted baseline so the sim still runs.
    """

    name = "llm"

    def __init__(self) -> None:
        self._fallback = ScriptedPolicy()

    def decide(self, usv: USV, world: World, dt_min: float, now_min: float) -> Vec2:
        # TODO: implement act(usv, obs, dt_min) with a real LLM decision and drop
        # this override. Placeholder = scripted baseline so the sim runs.
        return self._fallback.decide(usv, world, dt_min, now_min)


class MDPPolicy(Policy):
    """MDP-driven agent: the solved ENGAGEMENT MDP (reschu.mdp).

    MDP-mode USVs are NOT auto-identified: deciding WHEN to commit to a class,
    and WHICH class, is the MDP's job. A wrong declaration is the MDP's own
    error — nothing is injected.

    Per tick:
      * the team's cue history is updated from the truth-free Observation;
      * while the USV's assigned contact is undeclared, `act()` runs the
        engagement MDP (reschu.mdp.agent.EngagementAgent): belief from the fitted
        cue model, state = (distance band, belief cell), action from the solved
        policy table for this map size — approach / hold / back off / declare
        (a committed close-in from outside ID range) / abandon;
      * once the contact has been declared, the shared engagement mechanics
        (circle a perceived enemy, escort a perceived friendly, release a
        civilian) carry it out, exactly as for any other agent.

    WHICH contact each MDP USV works is the MDP TEAM ASSIGNMENT
    (mdp/assignment.py): once per tick, before any MDP USV moves, the
    coordinator re-matches the team when due (every 10 s, or at once when an MDP
    USV is free) using the solved MDP's values. It reads only own-side state
    from the world (our USVs' positions, assignments, modes, and whether their
    contacts are declared); everything about boats comes from the Observation.
    With `world.mdp_assignment == "nearest"` the engine's closest-unworked-boat
    rule picks instead (kept for comparison).

    FROZEN FOR TRIALS: on first use it verifies the policy files against
    mdp/FROZEN.json (mdp/frozen.py) and refuses to run if they changed. Every
    decision and every assignment change is recorded in `self.log`
    (mdp/decision_log.py), stamped with the policy version and fingerprint; the
    console writes it out with the results files.
    """

    name = "mdp"

    def __init__(self) -> None:
        self._agent = None
        self._assigner = None
        self._coord_t: float | None = None    # time of the last coordination
        self._world: World | None = None      # the mission this policy is serving
        from .mdp.decision_log import DecisionLog
        self.log = DecisionLog()
        self.version: dict = {}

    @property
    def agent(self):
        if self._agent is None:            # lazy: loads the cue model + tables
            from .mdp.agent import EngagementAgent, TableChooser
            from .mdp.frozen import check
            self.version = check()         # raises if frozen files were changed
            self.log.version = self.version["version"]
            self.log.fingerprint = self.version["fingerprint"]
            self._agent = EngagementAgent(TableChooser(), on_decision=self._on_decision)
        return self._agent

    def _on_decision(self, usv_id, boat_id, now_min, band, post, action) -> None:
        """Log one engagement decision with its values (what the agent knew)."""
        from .mdp.engagement import belief_cell, size_for_width, state_index
        from .mdp.features import id_range_for_width
        w = self._world
        width = w.grid.width if w is not None else 40.0
        table = self.agent.chooser.table(size_for_width(width))
        q = table.q[state_index(band, belief_cell(post))] if table.q else None
        st = self.agent._state.get(usv_id, {})
        dist = st.get("hold_d", float("nan")) / id_range_for_width(width)
        self.log.decision(now_min, usv_id, boat_id, band, dist, post, action, q)

    @property
    def assigner(self):
        if self._assigner is None:
            from .mdp.assignment import TeamAssigner
            self._assigner = TeamAssigner(self.agent, self.agent.chooser)
        return self._assigner

    def _coordinate(self, world: World, obs: Observation, dt_min: float) -> None:
        """Team assignment for all MDP USVs, once per tick (own-side state only)."""
        if world.mdp_assignment not in ("value", "rate"):
            return
        self.assigner.mode = world.mdp_assignment
        if self._coord_t is not None and abs(obs.now_min - self._coord_t) < 1e-12:
            return
        self._coord_t = obs.now_min
        from .mdp.assignment import TeamMember
        team = [u for u in world.usvs if u.mode is USVMode.MDP
                and not (u.rtb or u.left_grid or u.eliminated)]
        held = {u.assigned_target_id for u in world.usvs
                if u not in team and u.assigned_target_id is not None}
        members = []
        for u in team:
            tid = u.assigned_target_id
            bv = obs.boat(tid) if tid else None
            if bv is None or bv.left_grid:
                tid = None
            if tid is None:
                u.assignment_pinned = False
            locked = tid is not None and (bv.identified or u.assignment_pinned
                                          or self.agent.committed(u.id, tid))
            members.append(TeamMember(u.id, u.position, u.effective_speed_knots(), tid, locked))
            if locked:
                held.add(tid)
        if not self.assigner.due(obs.now_min, members, dt_min):
            return
        before = {m.id: m.assigned for m in members}
        for uid, bid in self.assigner.assign(members, obs, held, dt_min).items():
            world.usv(uid).assigned_target_id = bid
            old = before.get(uid)
            if bid != old:
                sc = self.assigner.last_scores.get(uid, {})
                self.log.assignment(obs.now_min, uid, old, bid, sc.get(bid), sc.get(old))

    def _new_mission(self, world: World) -> None:
        """A different World = a new mission: forget all per-mission state.
        (A reference is held, so a new world can never be mistaken for the old.)"""
        self._world = world
        self._coord_t = None
        self.log.clear()
        if self._agent is not None:
            self.agent.reset()
        if self._assigner is not None:
            self._assigner._last_t = None

    def decide(self, usv: USV, world: World, dt_min: float, now_min: float) -> Vec2:
        if world is not self._world:
            self._new_mission(world)
        obs = observe(usv, world, now_min)
        self.agent.track(obs)              # every tick, even while circling/escorting
        self._coordinate(world, obs, dt_min)
        obs.me.assigned_target_id = usv.assigned_target_id   # may have just changed
        tid = usv.assigned_target_id
        boat = world.boat(tid) if tid else None
        if boat is None or boat.left_grid or boat.identified:
            # declared (perceived class known) or nothing to work: mechanics only
            return assigned_agent_step(usv, world, dt_min, now_min)
        return self.act(usv, obs, dt_min)

    def act(self, usv: USV, obs: Observation, dt_min: float) -> Vec2:
        return self.agent.act(usv, obs, dt_min)


# One shared instance of each policy, chosen by the USV's mode.
_SCRIPTED = ScriptedPolicy()
_POLICIES: dict[USVMode, Policy] = {
    USVMode.LLM: LLMPolicy(),
    USVMode.MDP: MDPPolicy(),
    USVMode.GREEDY: _SCRIPTED,     # the greedy baseline (nearest boat); its wrong IDs: identification.py
}


def policy_for(usv: USV) -> Policy:
    """Return the policy that drives this (agent-mode) USV, based on its mode.
    Falls back to the internal scripted baseline for anything unexpected."""
    return _POLICIES.get(usv.mode, _SCRIPTED)
