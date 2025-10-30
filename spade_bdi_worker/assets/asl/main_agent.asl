// SPADE-BDI + Q-Learning Online Agent (bundled for refactored worker)
// ===================================================================
// This file mirrors the legacy AgentSpeak plans used by the original
// spadeBDI_RL project. It is packaged here so that subprocess workers
// can load the same plan library without depending on the legacy repo
// layout. Any updates should be mirrored in the original source until
// the refactor fully replaces the old stack.
// ===================================================================

// === INITIAL BELIEFS & GOALS ===
pos(0,0).
goal_pos(7,7).       // default; can be changed via .set_goal(Gx,Gy)
step_count(0).
!start.

// === DERIVED BELIEFS ===
at_goal :- pos(X,Y) & goal_pos(X,Y).

// === MAIN PLANS ===

@trigger_start_handler
+trigger_start <-
    -trigger_start;   // make it transient so we can trigger again later
    !start.

@start_plan
+!start <-
    .print("[AGENT] BDI Agent Started - With Policy Caching");
    // .clear_policy_store;      // uncomment for step 1 (fresh start)
    .reset_environment;
    -step_count(_); +step_count(0);  // ensure clean step count
    goal_pos(Gx,Gy);
    .check_cached_policy(Gx, Gy, 0.60);
    !navigate.

// ========================================================================
// ENHANCED NAVIGATION WITH CACHED POLICY REUSE
// ========================================================================

@use_cached
+!navigate : has_policy(PolicyName, SeqStr) & not at_goal <-
    .print("Using cached policy: ", PolicyName);
    .set_epsilon(0.0);
    .exec_cached_seq(SeqStr);
    !navigate.

@learn_online
+!navigate : not has_policy(_,_) & not at_goal & step_count(Steps) & Steps < 100 <-
    pos(X,Y); .get_state_from_pos(X,Y,S);
    .set_epsilon(0.1);
    .get_best_action(S, Action);
    .execute_action(Action);
    !navigate.

@max_steps_reached
+!navigate : step_count(Steps) & Steps >= 100 <-
    .print("Max steps reached (", Steps, "), restarting episode");
    -step_count(_); +step_count(0);
    !trigger_restart.

@success_goal_reached
+!navigate : at_goal <-
    pos(X,Y); step_count(Steps);
    .print("SUCCESS at (", X, ",", Y, ") in ", Steps, " steps");
    +goal_reached;              
    -step_count(_); +step_count(0);
    !trigger_restart.

// Fallback only if we truly lack position info
@fallback_plan
+!navigate : not pos(_,_) <-
    .print("Navigation fallback: no pos/2 belief. Reinitializing...");
    !start.

// ========================================================================
// POLICY EXTRACTION AND CACHING ON SUCCESS (step 3)
// ========================================================================

@on_success
+goal_reached <-
    .print("Success! Extracting and caching policy...");
    .rl_propose_seq(0);                  // fills proposed_seq/1 & seq_confidence/1
    proposed_seq(SeqStr); seq_confidence(C);
    goal_pos(Gx,Gy);
    .cache_policy(Gx, Gy, SeqStr, C, 0.60);
    .clear_episode_flags.

@on_failure
+fell_in_hole <-
    .print("Failed - fell in hole! Restarting training...");
    .remove_cached_policy;
    .clear_episode_flags;
    !start.

// ========================================================================
// POLICY REPAIR AND RESTART
// ========================================================================

@policy_repair
+cached_policy_failed <-
    .print("Cached policy failed! Falling back to online learning...");
    .remove_cached_policy;
    !navigate.

@restart_training
+!trigger_restart <-
    !start.
