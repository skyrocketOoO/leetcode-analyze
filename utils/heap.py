import heapq

items = [
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 0],
  [1, 0, 1, 0, 0],
  [1, 0, 1, 1, 0],
  [1, 1, 1, 1, 0],
  [1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [1, 0, 1, 0, 0],
  [0, 0, 0, 0, 1],
  [1, 0, 0, 1, 0],
  [0, 1, 0, 0, 0],
  [1, 0, 1, 0, 0],
  [1, 0, 1, 0, 0],
  [0, 1, 0, 0, 0],
  [1, 0, 0, 0, 0],
  [0, 1, 0, 1, 0],
  [0, 0, 1, 0, 0],
  [1, 1, 0, 0, 0],
]



import heapq

class CustomHeapItem:
    def __init__(self, value):
        self.value = value
    
    def __lt__(self, other):
        return self.value < other.value  # For min-heap

    def __repr__(self):
        return str(self.value)

class CustomHeap:
    def __init__(self):
        self.heap = []
        self.value_to_pos = {}  # Dictionary for fast lookups

    def push(self, value):
        # Insert item but do not record the position immediately
        item = CustomHeapItem(value)
        heapq.heappush(self.heap, item)
        
        # Rebuild the index after pushing to get the correct positions
        self._rebuild_value_to_pos()

    def update_value(self, old_value, new_value):
        # Find the position of the old value using the dictionary
        pos = self.value_to_pos.get(old_value)
        if pos is None:
            return  # Value not in heap
        
        # Update the value
        self.heap[pos].value = new_value
        
        # Update the dictionary to reflect the new value
        del self.value_to_pos[old_value]
        self.value_to_pos[new_value] = pos

        # Restore heap properties
        heapq._siftdown(self.heap, 0, pos)  # Fix the heap from pos upwards
        heapq._siftup(self.heap, pos)       # Fix the heap from pos downwards

    def pop(self):
        item = heapq.heappop(self.heap)
        del self.value_to_pos[item.value]  # Remove from dictionary
        self._rebuild_value_to_pos()  # Rebuild the dictionary after pop
        return item

    def _rebuild_value_to_pos(self):
        # Rebuild the value-to-position dictionary by scanning the heap
        self.value_to_pos = {item.value: i for i, item in enumerate(self.heap)}

    def __repr__(self):
        return repr(self.heap)

# Usage example:
custom_heap = CustomHeap()
custom_heap.push(5)
custom_heap.push(1)
custom_heap.push(3)

print("Heap before update:", custom_heap)

# Update value from 5 to 2
custom_heap.update_value(5, 2)

print("Heap after update:", custom_heap)

# Pop elements from heap
while custom_heap.heap:
    print("Popped:", custom_heap.pop())

