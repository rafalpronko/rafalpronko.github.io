"""Synthetic help-centre pages for a fictional SaaS product, plus query texts.

Written by hand for this experiment. No real product, user or company data.
Relevance labels live in labels.py; this module does not know them.
"""

DOCUMENTS: dict[str, list[str]] = {
    "export-api": [
        "The bulk export endpoint is designed for nightly batch jobs.",
        "It accepts at most 500 rows per call.",
        "Larger exports must be split into pages with the cursor parameter.",
        "Each page link expires after 15 minutes.",
    ],
    "search-api": [
        "The search endpoint serves interactive queries from the dashboard.",
        "It accepts at most 50 rows per call.",
        "Responses are cached for 30 seconds.",
    ],
    "errors": [
        "Error ERR-4012 means the API token has expired.",
        "Request a new token on the settings page to fix it.",
        "Error ERR-4013 means the API token lacks the export scope.",
        "Ask a workspace administrator to grant the missing scope.",
    ],
    "billing": [
        "Refunds for double billing are issued within five business days.",
        "The refund goes to the original payment card.",
        "Invoices are generated on the first day of each month.",
        "The business plan price is 30 euros per seat per month.",
    ],
    "retention": [
        "Inactive projects are deleted after 90 days.",
        "Owners receive a warning email two weeks before deletion.",
        "Deleted projects cannot be restored.",
    ],
    "faq": [
        "Customers often ask about the row limit of the bulk export endpoint.",
        "Customers also ask whether invoices can be split between teams.",
        "Both topics are covered on the reference pages.",
        "Contact support if the reference pages do not answer your question.",
    ],
    "security": [
        "API tokens expire after 30 days by default.",
        "Administrators can shorten this period in the security settings.",
        "Expired tokens return an authentication error.",
    ],
    "team-plan": [
        "The team plan is aimed at groups of up to twenty people.",
        "It includes shared dashboards.",
        "It also includes single sign-on.",
        "Audit logs are kept for one year.",
        "Its price is 12 euros per seat per month.",
    ],
}

QUERIES: dict[str, str] = {
    "q01": "What is the row limit of the bulk export endpoint?",
    "q02": "How many rows can the search endpoint accept per call?",
    "q03": "What does error ERR-4013 mean?",
    "q04": "How soon can I expect a reimbursement for a duplicate charge?",
    "q05": "Where does the refund money go?",
    "q06": "When do API tokens expire?",
    "q07": "What is the price per seat of the team plan?",
    "q08": "How long is the audit trail retained?",
    "q09": "Can a deleted project be recovered?",
    "q10": "When are dormant workspaces purged?",
}
