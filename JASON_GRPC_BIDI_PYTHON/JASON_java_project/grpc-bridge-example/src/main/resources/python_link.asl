!bootstrap.

+!bootstrap <-
    .print("[Jason] requesting first action...");
    request_action("startup");
    push_percept("temperature=32C").

+server_action(Action, Metadata) <-
    .print("[Jason] python suggests:", Action, "info:", Metadata);
    .wait(1000);
    request_action("search").
