"""
Example 14: Advanced RunnableBranch - Dynamic Workflow Routing
============================================================

This example demonstrates how RunnableBranch can be used for dynamic workflow routing, such as routing support tickets by topic or urgency.
"""

from langchain_core.runnables import RunnableLambda, RunnableBranch

# Step 1: Classify ticket topic
def classify_topic(ticket):
    text = ticket["text"].lower()
    if "payment" in text or "invoice" in text:
        return "billing"
    if "error" in text or "bug" in text:
        return "technical"
    if "feature" in text or "request" in text:
        return "feature"
    return "general"

# Step 2: Classify urgency
def is_urgent(ticket):
    return "urgent" in ticket["text"].lower() or ticket.get("priority", "") == "high"

def handle_billing(ticket):
    return f"[Billing] Ticket routed: {ticket['text']}"

def handle_technical(ticket):
    return f"[Technical] Ticket routed: {ticket['text']}"

def handle_feature(ticket):
    return f"[Feature] Ticket routed: {ticket['text']}"

def handle_general(ticket):
    return f"[General] Ticket routed: {ticket['text']}"

def handle_urgent(ticket):
    return f"[URGENT] Immediate attention: {ticket['text']}"

if __name__ == "__main__":
    # First, check for urgency
    urgency_branch = RunnableBranch(
        (is_urgent, RunnableLambda(handle_urgent)),
        (lambda t: True, RunnableLambda(lambda t: t)),  # fallback: pass ticket along
    )
    # Then, route by topic
    topic_branch = RunnableBranch(
        (lambda t: classify_topic(t) == "billing", RunnableLambda(handle_billing)),
        (lambda t: classify_topic(t) == "technical", RunnableLambda(handle_technical)),
        (lambda t: classify_topic(t) == "feature", RunnableLambda(handle_feature)),
        (lambda t: True, RunnableLambda(handle_general)),
    )
    # Compose: urgency check -> topic routing
    def workflow(ticket):
        result = urgency_branch.invoke(ticket)
        if isinstance(result, dict):  # not urgent, route by topic
            return topic_branch.invoke(result)
        return result

    tickets = [
        {"text": "I have an urgent payment issue!", "priority": "high"},
        {"text": "There is a bug in the system."},
        {"text": "Can you add this feature request?"},
        {"text": "How do I change my password?"},
    ]
    print("\n=== Advanced RunnableBranch: Support Ticket Routing ===")
    for t in tickets:
        print(f"Ticket: {t['text']}")
        print("Routed to:", workflow(t))
        print("-") 