import os
import json
import random

RAW_DATA_PATH = "data/raw/customer_support_raw.jsonl"

def generate_raw_dataset():
    os.makedirs("data/raw", exist_ok=True)
    
    # Representative baseline templates with realistic noise, PII, and messy variations
    tickets = [
        # Billing
        {
            "ticket_id": "TICK-1001",
            "customer_id": "CUST-8832",
            "customer_name": "Alice Smith",
            "email": "alice.smith@example.com",
            "phone": "+1-555-0192",
            "category": "billing_inquiry",
            "difficulty": "easy",
            "customer_message": "Hi, I was charged twice for my subscription this month ($29.99 x 2). Can you check invoice #INV-9041?",
            "agent_response": "I apologize for the double charge. I have verified invoice #INV-9041 and processed a refund of $29.99 back to your original payment method within 3-5 business days.",
            "should_escalate": False,
            "required_facts": ["duplicate charge of $29.99 verified", "refund processed to original payment method in 3-5 days"],
            "noise_level": "low"
        },
        {
            "ticket_id": "TICK-1002",
            "customer_id": "CUST-4411",
            "customer_name": "Bob Jones",
            "email": "bob.j@gmail.com",
            "phone": "555-0143",
            "category": "billing_inquiry",
            "difficulty": "medium",
            "customer_message": "WHY ARE THERE EXTRA UNEXPLAINED FEES ON MY BILL??!! INV-8812 shows $45 instead of $30!! FIX THIS NOW OR I CANCEL EVERYTHING!!!",
            "agent_response": "I understand your frustration. The extra $15 was an auto-renew add-on feature. I can remove this add-on and credit your account $15.",
            "should_escalate": False,
            "required_facts": ["$15 extra fee was due to auto-renew add-on", "add-on removed and $15 credit applied"],
            "noise_level": "high_caps"
        },
        {
            "ticket_id": "TICK-1003",
            "customer_id": "CUST-9012",
            "customer_name": "Charles Vance",
            "email": "cvance@corp.org",
            "phone": "N/A",
            "category": "billing_inquiry",
            "difficulty": "hard",
            "customer_message": "Our corporate tax-exempt account was charged VAT taxes across 15 separate transactions last quarter totaling $4,250. We need tax exemption re-applied retroactively and wire credit back to our banking partner.",
            "agent_response": "Due to retroactive corporate tax exemption and wire transfer adjustments, I am escalating your ticket to our specialized Billing & Finance Compliance team.",
            "should_escalate": True,
            "required_facts": ["escalated to Billing & Finance Compliance team", "requires corporate tax exemption verification and wire adjustment"],
            "noise_level": "medium"
        },
        # Account Access
        {
            "ticket_id": "TICK-1004",
            "customer_id": "CUST-1049",
            "customer_name": "David Miller",
            "email": "dave_m@yahoo.com",
            "phone": "+1-555-9981",
            "category": "account_access",
            "difficulty": "easy",
            "customer_message": "Forgot my password and the reset link emailed to me expired. How do I get a new one?",
            "agent_response": "You can request a new password reset link by visiting the login page and clicking 'Forgot Password'. Make sure to click the link within 15 minutes.",
            "should_escalate": False,
            "required_facts": ["visit login page and click Forgot Password", "reset link expires in 15 minutes"],
            "noise_level": "low"
        },
        {
            "ticket_id": "TICK-1005",
            "customer_id": "CUST-5521",
            "customer_name": "Eva Green",
            "email": "eva.green@domain.net",
            "phone": "555-2231",
            "category": "account_access",
            "difficulty": "hard",
            "customer_message": "My 2FA phone number was changed by a hacker! I'm locked out of my account and someone is making unauthorized changes right now! GET ME A HUMAN MANAGER IMMEDIATELY!",
            "agent_response": "I am escalating this security emergency immediately to our Security Escalation Team to freeze account activity and initiate identity verification.",
            "should_escalate": True,
            "required_facts": ["escalated immediately to Security Escalation Team", "account frozen pending identity verification"],
            "noise_level": "high_urgency"
        },
        # Technical Issue
        {
            "ticket_id": "TICK-1006",
            "customer_id": "CUST-7741",
            "customer_name": "Frank White",
            "email": "fwhite@test.com",
            "phone": "555-8822",
            "category": "technical_issue",
            "difficulty": "easy",
            "customer_message": "Getting error 504 Gateway Timeout when trying to export PDF reports on desktop app.",
            "agent_response": "Error 504 indicates a temporary backend timeout during high server load. Please clear your cache or try again in a few minutes. If it persists, ensure app version is updated to v2.4.",
            "should_escalate": False,
            "required_facts": ["Error 504 is a temporary server timeout", "clear cache or retry in a few minutes", "ensure v2.4 update"],
            "noise_level": "low"
        },
        {
            "ticket_id": "TICK-1007",
            "customer_id": "CUST-1102",
            "customer_name": "Grace Hopper",
            "email": "grace@navy.mil",
            "phone": "555-0001",
            "category": "technical_issue",
            "difficulty": "medium",
            "customer_message": "App crashes immediately upon launch on iOS 18.1 after update. Reinstalling didn't work.",
            "agent_response": "This issue occurs on iOS 18.1 with dark mode active. As a temporary workaround, toggle off dark mode in system settings while our team prepares hotfix v2.4.1.",
            "should_escalate": False,
            "required_facts": ["workaround is toggling off dark mode in iOS system settings", "hotfix v2.4.1 coming soon"],
            "noise_level": "low"
        },
        # Order & Shipping
        {
            "ticket_id": "TICK-1008",
            "customer_id": "CUST-3390",
            "customer_name": "Hannah Abbott",
            "email": "hannah.a@gmail.com",
            "phone": "555-7711",
            "category": "order_shipping",
            "difficulty": "easy",
            "customer_message": "Where is my order #ORD-55412? Tracking status says 'In Transit' for 3 days.",
            "agent_response": "Your order #ORD-55412 is currently in transit with FedEx (Tracking #FX-99214). Estimated delivery date is tomorrow by 5:00 PM.",
            "should_escalate": False,
            "required_facts": ["order ORD-55412 tracked via FedEx FX-99214", "estimated delivery tomorrow by 5 PM"],
            "noise_level": "low"
        },
        {
            "ticket_id": "TICK-1009",
            "customer_id": "CUST-6621",
            "customer_name": "Ian Malcolm",
            "email": "imalcolm@dino.com",
            "phone": "555-4433",
            "category": "order_shipping",
            "difficulty": "medium",
            "customer_message": "Package delivered to wrong address! Tracking says delivered to 123 Main St, but I live at 456 Elm St!",
            "agent_response": "I apologize for the misdelivery. I have filed a carrier investigation with FedEx and issued a free replacement order #ORD-55990 shipped via expedited delivery.",
            "should_escalate": False,
            "required_facts": ["carrier investigation filed", "free replacement order ORD-55990 issued"],
            "noise_level": "medium"
        },
        # Refund Request
        {
            "ticket_id": "TICK-1010",
            "customer_id": "CUST-8819",
            "customer_name": "Julia Roberts",
            "email": "jules@hollywood.com",
            "phone": "555-3344",
            "category": "refund_request",
            "difficulty": "easy",
            "customer_message": "I ordered 3 days ago and want to return an unopened item within 30 days return window. How?",
            "agent_response": "You can return unopened items within 30 days. Go to Orders -> Select Order -> Click 'Initiate Return' to download your prepaid shipping label.",
            "should_escalate": False,
            "required_facts": ["30-day return policy for unopened items", "download shipping label under Orders -> Initiate Return"],
            "noise_level": "low"
        },
        # Policy Exception
        {
            "ticket_id": "TICK-1011",
            "customer_id": "CUST-9901",
            "customer_name": "Kevin Bacon",
            "email": "kbacon@actor.com",
            "phone": "555-9090",
            "category": "policy_exception",
            "difficulty": "hard",
            "customer_message": "My purchase was 90 days ago (past the 30 day return limit), but the product broke due to a medical emergency while I was out of country. Requesting full cash refund override.",
            "agent_response": "As this request is outside our standard 30-day return policy and requires an exception override, I am escalating your request to a Senior Support Supervisor.",
            "should_escalate": True,
            "required_facts": ["request is outside standard 30-day policy", "escalated to Senior Support Supervisor for exception review"],
            "noise_level": "medium"
        },
        # General Info
        {
            "ticket_id": "TICK-1012",
            "customer_id": "CUST-1212",
            "customer_name": "Laura Palmer",
            "email": "laura@twinpeaks.org",
            "phone": "555-1212",
            "category": "general_info",
            "difficulty": "easy",
            "customer_message": "What are your customer support operating hours?",
            "agent_response": "Our support team is available 24/7 via live chat and email support. Phone support is available Monday through Friday, 9:00 AM - 6:00 PM EST.",
            "should_escalate": False,
            "required_facts": ["live chat and email available 24/7", "phone support Mon-Fri 9am-6pm EST"],
            "noise_level": "low"
        }
    ]
    
    expanded_tickets = []
    
    # Exact duplicate records (for duplicate audit test)
    expanded_tickets.append(tickets[0].copy())
    expanded_tickets.append(tickets[3].copy())
    
    # Malformed record (empty message)
    expanded_tickets.append({
        "ticket_id": "TICK-9999",
        "customer_id": "",
        "customer_name": "",
        "email": "invalid_email_format",
        "phone": "",
        "category": "unknown",
        "difficulty": "medium",
        "customer_message": "",
        "agent_response": "",
        "should_escalate": False,
        "required_facts": [],
        "noise_level": "malformed"
    })
    
    random.seed(42)
    counter = 2000
    for idx in range(300):
        base = random.choice(tickets)
        counter += 1
        t_id = f"TICK-{counter}"
        
        msg = base["customer_message"]
        rnd = random.random()
        if rnd < 0.2:
            msg = msg.lower()
        elif rnd < 0.35:
            msg = msg + " thx PLZ HELP FAST!!"
            
        rec = {
            "ticket_id": t_id,
            "customer_id": base["customer_id"],
            "customer_name": base["customer_name"],
            "email": base["email"],
            "phone": base["phone"],
            "category": base["category"],
            "difficulty": base["difficulty"],
            "customer_message": msg,
            "agent_response": base["agent_response"],
            "should_escalate": base["should_escalate"],
            "required_facts": base["required_facts"],
            "noise_level": base["noise_level"]
        }
        expanded_tickets.append(rec)
        
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        for t in expanded_tickets:
            f.write(json.dumps(t) + "\n")
            
    print(f"Generated raw dataset with {len(expanded_tickets)} records at {RAW_DATA_PATH}")

if __name__ == "__main__":
    generate_raw_dataset()
