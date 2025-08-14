# Implementing a "Record-Breaking" Bounded Multi-Source Shortest Path Algorithm

![GitHub language count](https://img.shields.io/github/languages/count/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm?style=for-the-badge)
![GitHub top language](https://img.shields.io/github/languages/top/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm?style=for-the-badge)

This repository contains a Python implementation and analysis of a novel **Bounded Multi-Source Shortest Path (BMSSP)** algorithm. The project started with a viral image of a research paper claiming to have developed the "best shortest-path algorithm in 41 years."

The goal of this project was to bridge the gap between a dense academic paper and a working, understandable program, and to analyze its real-world performance against classic algorithms.

---

## 📝 The Algorithm Explained

The BMSSP algorithm is designed for a specific and powerful use case: finding the shortest paths from multiple starting points (`Multi-Source`) but only up to a certain maximum distance or cost (`Bounded`).

### Core Concepts

* **The Strategy: Divide and Conquer**: Instead of exploring the graph one node at a time like traditional algorithms (e.g., Dijkstra's), BMSSP intelligently explores entire regions. It identifies a promising, unexplored node, recursively solves the shortest-path problem in a smaller "bubble" around it, and then merges those findings back into the main solution.

* **The Brain (`D`): A Specialized Priority Queue**: The core of the algorithm's theoretical efficiency comes from a special bucket-based priority queue. Instead of a simple sorted list, it groups nodes into "buckets" based on their distance (e.g., all nodes with distances in the range `[16, 31]` go into one bucket).

* **The Superpower: Batch Updates**: This bucket structure allows for a massive efficiency gain through `BATCHDECREASEKEY`. When the algorithm discovers thousands of new paths at once, it can re-categorize all of them in a single, efficient batch operation instead of updating the queue one by one.

---

## 📂 Project Structure

Here's how the repository is organized:

```
/
├── bmssp_algorithm.py      # The main implementation of the BMSSP algorithm.
├── baseline_dijkstra.py    # A standard bounded Dijkstra's algorithm for comparison.
├── benchmark.py            # Script to run a performance benchmark between the two algorithms.
├── test_pq.py              # Unit tests for the specialized priority queue.
└── README.md               # You are here.
```

---

## 🛠️ Setup and Installation

The project is written in standard Python 3. No external libraries are required.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm.git](https://github.com/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm.git)
    ```
2.  Navigate into the project directory:
    ```bash
    cd Bounded-Multi-Source-Shortest-Path-Algorithm
    ```

---

## 🚀 How to Run

You can run three main scripts: the unit tests, the main algorithm demonstration, and the performance benchmark.

### 1. Run the Unit Tests

To verify that the core `SpecializedPriorityQueue` data structure is working correctly:
```bash
python test_pq.py
```

### 2. Run the BMSSP Algorithm

To see the final, recursive BMSSP algorithm run on a small sample graph and print its output:
```bash
python bmssp_algorithm.py
```

### 3. Run the Performance Benchmark

To generate a large, sparse graph and compare the performance of `BMSSP` against the baseline `Dijkstra's`:
```bash
python benchmark.py
```

---

## 📊 Results and Analysis: Theory vs. Reality

The benchmark results highlight a crucial lesson in computer science.

**Finding**: When running the benchmark on a moderately sized graph (e.g., 1000 nodes), the classic Dijkstra's algorithm is significantly faster than our BMSSP implementation.

**Analysis**: This is the expected outcome and the core finding of this project.
* **Overhead**: The BMSSP algorithm, with its complex data structures, recursive calls, and multiple layers of logic, has a high "overhead" in a high-level language like Python.
* **Asymptotic Advantage**: The theoretical "record-breaking" speed of the paper only manifests at an immense scale (likely millions or billions of nodes) where the benefits of its intelligent "divide and conquer" strategy begin to outweigh its high constant costs.
* **Practicality**: For most common, real-world applications on consumer hardware, a simpler, more direct algorithm like Dijkstra's is often the more practical and faster choice.

---

## 🏁 Conclusion

This project successfully translated a complex, theoretical algorithm into a working, understandable Python implementation. It serves as a practical exploration of the gap between theoretical computer science and software engineering.

The journey confirms that while academic breakthroughs push the boundaries of what's possible, the "best" algorithm for a task always depends on the specific context, scale, and practical constraints of the problem at hand.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.