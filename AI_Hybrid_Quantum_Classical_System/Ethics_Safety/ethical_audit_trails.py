import logging
from logging.handlers import TimedRotatingFileHandler
import json

def setup_logger():
    """ Set up logging configuration. """
    logger = logging.getLogger('EthicalAudit')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler('ethical_audit.log', when='midnight', backupCount=30)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class EthicalAuditTrail:
    def __init__(self):
        self.logger = setup_logger()

    def log_decision(self, decision_data):
        """ Log an ethical decision with detailed context. """
        # The decision_data should be a dictionary containing all relevant data for the decision
        self.logger.info(json.dumps(decision_data))

    def audit_decision(self, decision, context, user_id, rationale):
        """ Audit a decision for ethical compliance and log the audit trail. """
        audit_data = {
            'decision': decision,
            'context': context,
            'user_id': user_id,
            'rationale': rationale,
            'timestamp': str(datetime.now())
        }
        self.log_decision(audit_data)

# Example usage
if __name__ == '__main__':
    auditor = EthicalAuditTrail()
    decision_info = {
        'decision': 'Loan Approval',
        'context': 'Loan application from high-risk category',
        'user_id': 'user123',
        'rationale': 'Based on credit score and repayment history'
    }
    auditor.audit_decision(**decision_info)
