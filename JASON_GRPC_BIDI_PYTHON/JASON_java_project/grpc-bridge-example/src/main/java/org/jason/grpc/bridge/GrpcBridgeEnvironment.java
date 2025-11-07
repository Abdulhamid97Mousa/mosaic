package org.jason.grpc.bridge;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import jason.asSyntax.Literal;
import jason.asSyntax.StringTerm;
import jason.asSyntax.Structure;
import jason.asSyntax.Term;
import jason.environment.Environment;
import org.jason.grpc.proto.ActionRequest;
import org.jason.grpc.proto.ActionResponse;
import org.jason.grpc.proto.AgentBridgeGrpc;
import org.jason.grpc.proto.Percept;

/** Environment that proxies custom AgentSpeak actions to the Python gRPC service. */
public class GrpcBridgeEnvironment extends Environment {

    private static final Logger LOG = Logger.getLogger(GrpcBridgeEnvironment.class.getName());

    private ManagedChannel channel;
    private AgentBridgeGrpc.AgentBridgeBlockingStub stub;

    @Override
    public void init(String[] args) {
        super.init(args);
        String target = (args != null && args.length > 0) ? args[0] : "localhost:50051";
        // TODO: Replace plaintext with TLS for non-localhost targets. For now kept for dev simplicity.
        channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        stub = AgentBridgeGrpc.newBlockingStub(channel);
        LOG.info(() -> "Connected to AgentBridge at " + target);
    }

    @Override
    public boolean executeAction(String agName, Structure action) {
        Objects.requireNonNull(agName, "agent name");
        Objects.requireNonNull(action, "action");
        String functor = action.getFunctor();
        if ("push_percept".equals(functor)) {
            handlePushPercept(agName, action);
            return true;
        }
        if ("request_action".equals(functor)) {
            handleRequestAction(agName, action);
            return true;
        }
        return super.executeAction(agName, action);
    }

    private void handlePushPercept(String agName, Structure action) {
        ensureArity(action, 1);
        String payload = termToString(action.getTerm(0));
        try {
            stub.pushPercept(Percept.newBuilder().setAgent(agName).setPayload(payload).build());
            LOG.info(() -> String.format("[%s] -> python percept: %s", agName, payload));
        } catch (StatusRuntimeException ex) {
            LOG.log(Level.SEVERE, "Failed to push percept", ex);
        }
    }

    private void handleRequestAction(String agName, Structure action) {
        ensureArity(action, 1);
        String context = termToString(action.getTerm(0));
        try {
            ActionResponse response = stub.requestAction(
                    ActionRequest.newBuilder().setAgent(agName).setContext(context).build());
            LOG.info(() -> String.format("[%s] <- python action: %s", agName, response.getAction()));
            sendServerActionPercept(agName, response);
        } catch (StatusRuntimeException ex) {
            LOG.log(Level.SEVERE, "Failed to request action", ex);
        }
    }

    private void sendServerActionPercept(String agName, ActionResponse response) {
        clearPercepts(agName);

        String actionEsc = esc(response.getAction());
        String metaEsc;
        if (!response.getMetaList().isEmpty()) {
            metaEsc = response.getMetaList().stream()
                    .map(e -> esc(e.getKey()) + "=" + esc(e.getValue()))
                    .collect(Collectors.joining(";"));
        } else {
            metaEsc = esc(response.getMetadata());
        }

        // Include status if present and not OK/UNSPECIFIED
        String statusPart = "";
        switch (response.getStatus()) {
            case ACTION_STATUS_ERROR:
                statusPart = ";status=error" + (response.getErrorDetail().isBlank() ? "" : ";detail=" + esc(response.getErrorDetail()));
                break;
            case ACTION_STATUS_PARTIAL:
                statusPart = ";status=partial";
                break;
            default:
                break; // OK or UNSPECIFIED -> no extra annotation
        }

        String combinedMeta = metaEsc + statusPart;
        String literal = String.format("server_action(\"%s\", \"%s\")", actionEsc, combinedMeta);
        try {
            addPercept(agName, Literal.parseLiteral(literal));
        } catch (Exception ex) {
            LOG.log(Level.SEVERE, "Failed to parse server_action literal: " + literal, ex);
        }
    }

    private static String esc(String s) {
        if (s == null) return "";
        // Escape backslashes and quotes; remove newlines to keep one-line literal.
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replaceAll("[\r\n]", " ");
    }

    private static void ensureArity(Structure structure, int expected) {
        if (structure.getArity() != expected) {
            throw new IllegalArgumentException(
                    String.format("Action %s expects %d argument(s)", structure.getFunctor(), expected));
        }
    }

    private static String termToString(Term term) {
        if (term.isString()) {
            return ((StringTerm) term).getString();
        }
        return term.toString();
    }

    @Override
    public void stop() {
        super.stop();
        if (channel != null) {
            channel.shutdown();
            try {
                channel.awaitTermination(3, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
