package org.jason.grpc.bridge;

import jason.infra.local.RunLocalMAS;

/** Thin launcher so we can run the MAS via `./gradlew :grpc-bridge-example:run`. */
public final class RunBridgeMas {

    private RunBridgeMas() {}

    public static void main(String[] args) throws Exception {
        String masPath = args.length > 0 ? args[0] : "src/main/resources/grpc_bridge.mas2j";
        RunLocalMAS.main(new String[] {masPath});
    }
}
