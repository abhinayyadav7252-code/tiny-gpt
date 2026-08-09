import json
import os

class WorkingMemory:
    def __init__(self):
        self.context = []

    def add(self, role, content):
        self.context.append({"role": role, "content": content})

    def get_context_string(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.context])

class LongTermMemory:
    def __init__(self, file_path='memory.json'):
        self.file_path = file_path
        self.facts = []
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.facts = data.get("facts", [])
        else:
            self.facts = []
            self.save()

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump({"facts": self.facts}, f, indent=2)

    def add_fact(self, key, value):
        # Update existing fact if key matches, else append
        for fact in self.facts:
            if fact["key"] == key:
                fact["value"] = value
                self.save()
                return
        self.facts.append({"key": key, "value": value})
        self.save()

    def get_all_facts(self):
        return self.facts
