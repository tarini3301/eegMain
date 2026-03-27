"""
NeuroAge Task Queue
===================
A simple in-memory task manager for handling high-latency asynchronous jobs.
Supports:
1. Job enqueuing.
2. Background execution via threading.
3. Status polling and result retrieval.
"""

import uuid
import threading
import time
from datetime import datetime

class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TaskManager, cls).__new__(cls)
                cls._instance.tasks = {}
        return cls._instance

    def create_task(self, func, *args, **kwargs):
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            'status': 'pending',
            'progress': 0,
            'message': 'Queued',
            'result': None,
            'error': None,
            'created_at': datetime.now().isoformat()
        }
        
        # Start background thread
        thread = threading.Thread(target=self._run_task, args=(task_id, func, args, kwargs))
        thread.daemon = True
        thread.start()
        
        return task_id

    def _run_task(self, task_id, func, args, kwargs):
        try:
            self.tasks[task_id]['status'] = 'running'
            self.tasks[task_id]['message'] = 'Analysis started...'
            self.tasks[task_id]['progress'] = 10
            
            # Execute original function — pass task_id as first arg for progress updates
            result = func(task_id, *args, **kwargs)
            
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['progress'] = 100
            self.tasks[task_id]['message'] = 'Analysis complete'
            self.tasks[task_id]['result'] = result
            
        except Exception as e:
            self.tasks[task_id]['status'] = 'failed'
            self.tasks[task_id]['error'] = str(e)
            self.tasks[task_id]['message'] = f"Error: {str(e)}"

    def get_status(self, task_id):
        return self.tasks.get(task_id)

    def update_progress(self, task_id, progress, message=None):
        if task_id in self.tasks:
            self.tasks[task_id]['progress'] = progress
            if message:
                self.tasks[task_id]['message'] = message

# Global instance
tasks = TaskManager()
