# Smart-Task-Scheduler

Creating a Smart-Task-Scheduler in Python involves several components including task input, a machine learning model for optimization, and scheduling logic. Below, I'll provide a simplified version of such a program, which schedules tasks based on deadlines, dependencies, and user-defined priorities. Note that for a real-world application, a more intricate machine learning model would be necessary, potentially with the use of libraries like TensorFlow or PyTorch.

The following code focuses on using a simple priority-based scheduler with basic ML-inspired sorting logic, given the complexity of implementing a full ML model in this format:

```python
import heapq

class Task:
    def __init__(self, name, deadline, priority, dependencies=[]):
        """Initialize a task with a name, deadline, priority, and optional dependencies."""
        self.name = name
        self.deadline = deadline
        self.priority = priority
        self.dependencies = set(dependencies)
        self.state = 'pending'  # can be 'pending', 'in-progress', or 'completed'
    
    def is_ready_to_schedule(self, completed_tasks):
        """Check if all dependencies of the task have been completed."""
        return self.dependencies.issubset(completed_tasks)


class SmartTaskScheduler:
    def __init__(self):
        """Initialize the scheduler with an empty task list."""
        self.tasks = []
    
    def add_task(self, task):
        """Add a task to the scheduler."""
        if not isinstance(task, Task):
            raise ValueError("Invalid task. Must be of type Task.")
        self.tasks.append(task)
    
    def schedule_tasks(self):
        """Schedule tasks to maximize productivity."""
        completed_tasks = set()
        ready_queue = []

        # Tasks that are ready to be scheduled will be pushed into the heap
        for task in self.tasks:
            if task.is_ready_to_schedule(completed_tasks):
                heapq.heappush(ready_queue, (-task.priority, task.deadline, task))

        schedule_order = []

        while ready_queue:
            # Remove the task with the highest priority (smallest negative value)
            _, _, task = heapq.heappop(ready_queue)
            task.state = 'in-progress'
            schedule_order.append(task)

            task.state = 'completed'
            completed_tasks.add(task.name)
            
            # Add new ready tasks to the heap
            for t in self.tasks:
                if t.state == 'pending' and t.is_ready_to_schedule(completed_tasks):
                    heapq.heappush(ready_queue, (-t.priority, t.deadline, t))

        return schedule_order

    def show_schedule(self, schedule_order):
        """Display the task schedule."""
        for task in schedule_order:
            print(f"Task: {task.name}, Priority: {task.priority}, Deadline: {task.deadline}, State: {task.state}")

def main():
    # Demonstration of how the Smart Task Scheduler works

    task1 = Task(name='Task 1', deadline=5, priority=3)
    task2 = Task(name='Task 2', deadline=2, priority=4, dependencies=['Task 1'])
    task3 = Task(name='Task 3', deadline=4, priority=2, dependencies=['Task 1'])
    task4 = Task(name='Task 4', deadline=1, priority=5)

    scheduler = SmartTaskScheduler()

    try:
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        scheduler.add_task(task4)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        schedule_order = scheduler.schedule_tasks()
        scheduler.show_schedule(schedule_order)
    except Exception as e:
        print(f"An error occurred while scheduling the tasks: {e}")

if __name__ == "__main__":
    main()
```

### Explanation:
1. **Task Class**: Represents a task with a deadline, priority, and dependencies. It also checks if dependencies are resolved.
  
2. **SmartTaskScheduler Class**: Manages task scheduling. It uses a priority queue (heap) to schedule tasks based on descending priority order, and tasks are re-checked for being ready once their dependencies are met.

3. **Error Handling**: Includes basic type checking when adding tasks and exception handling while scheduling tasks.

This program uses priority-based sorting with dependencies to mimic task optimization. A true model would require training data showing past task completions and patterns to learn effectively, which is not captured here.