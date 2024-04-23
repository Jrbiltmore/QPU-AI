
# /User_Interface_Experience/feedback_system.py

class FeedbackSystem:
    def __init__(self):
        self.feedback_storage = []

    def collect_feedback(self, feedback):
        self.feedback_storage.append(feedback)
        self.analyze_feedback()

    def analyze_feedback(self):
        # Analyze the collected feedback for trends, sentiment, etc.
        # Placeholder for actual analysis logic
        positive = sum(1 for f in self.feedback_storage if 'good' in f.lower())
        negative = sum(1 for f in self.feedback_storage if 'bad' in f.lower())
        return {
            'total_feedback': len(self.feedback_storage),
            'positive': positive,
            'negative': negative
        }

# Example usage
if __name__ == '__main__':
    system = FeedbackSystem()
    system.collect_feedback('This product is really good!')
    system.collect_feedback('This product is bad and needs improvement.')
    results = system.analyze_feedback()
    print(results)
