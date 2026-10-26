# sample_data.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: High-quality sample educational lecture data for PrepGenius.
#          Allows running the full pipeline in Demo Mode without requiring
#          any OpenAI or Gemini API keys.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

SAMPLE_LECTURE_TITLE = "MIT 6.006: Introduction to Algorithms — Asymptotic Analysis & Binary Search"

SAMPLE_TRANSCRIPT_SEGMENTS: list[dict[str, Any]] = [
    {"start": 0.0, "duration": 4.5, "text": "Welcome to MIT 6.006: Introduction to Algorithms."},
    {"start": 4.5, "duration": 5.0, "text": "Today we begin our study of algorithmic thinking and computational complexity."},
    {"start": 9.5, "duration": 6.2, "text": "An algorithm is a well-defined computational procedure that takes inputs and produces outputs."},
    {"start": 15.7, "duration": 5.8, "text": "We evaluate algorithms primarily by their asymptotic running time as input size n tends to infinity."},
    {"start": 21.5, "duration": 6.0, "text": "Big-O notation gives an asymptotic upper bound: f(n) = O(g(n)) if f(n) <= c * g(n) for large n."},
    {"start": 27.5, "duration": 5.5, "text": "Big-Omega gives a lower bound, and Big-Theta represents an asymptotically tight bound."},
    {"start": 33.0, "duration": 6.5, "text": "Let us consider the problem of searching for a key x in a sorted array A of length n."},
    {"start": 39.5, "duration": 5.5, "text": "Linear search checks every single element sequentially from index 0 to n minus 1."},
    {"start": 45.0, "duration": 5.0, "text": "The worst-case running time of linear search is clearly Theta of n comparisons."},
    {"start": 50.0, "duration": 6.0, "text": "However, because the array is sorted, we can exploit random access with binary search."},
    {"start": 56.0, "duration": 6.5, "text": "Binary search compares the target x to the middle element A[mid] where mid is low plus high over 2."},
    {"start": 62.5, "duration": 5.5, "text": "If x equals A[mid], we return mid immediately. If x is smaller, search the left half."},
    {"start": 68.0, "duration": 6.0, "text": "If x is larger, we recursively search the right half between mid plus 1 and high."},
    {"start": 74.0, "duration": 6.5, "text": "This leads to the recurrence relation: T(n) = T(n / 2) + O(1)."},
    {"start": 80.5, "duration": 6.0, "text": "At each recursive level, we perform constant work and halve the remaining search range."},
    {"start": 86.5, "duration": 6.5, "text": "Solving this recurrence using the Master Theorem gives a worst-case time complexity of O(log n)."},
    {"start": 93.0, "duration": 6.0, "text": "Notice that log base 2 of one billion is only approximately thirty comparisons!"},
    {"start": 99.0, "duration": 6.5, "text": "In comparison, linear search could require one billion comparisons in the worst case."},
    {"start": 105.5, "duration": 6.0, "text": "The critical prerequisite is that the array must remain strictly sorted beforehand."},
    {"start": 111.5, "duration": 6.0, "text": "If the array is unsorted, sorting it first takes O(n log n), so binary search pays off for repeated queries."},
    {"start": 117.5, "duration": 5.5, "text": "For your homework, implement both iterative and recursive versions and measure the call stack depth."},
    {"start": 123.0, "duration": 5.0, "text": "Next lecture, we will examine divide-and-conquer sorting algorithms including Merge Sort."},
]

SAMPLE_REVISION_NOTES: str = """# MIT 6.006: Introduction to Algorithms — Asymptotic Analysis & Binary Search

## 1. Lecture Overview
This lecture introduces asymptotic runtime analysis (Big-O, Big-Omega, Big-Theta) and examines binary search on sorted arrays as a canonical example of divide-and-conquer logarithmic efficiency.

## 2. Key Concepts
- **Asymptotic Analysis**: Evaluating algorithm scalability as input size $n \\to \\infty$ without hardware-specific constants.
- **Asymptotic Bounds Hierarchy**:
  - $O(g(n))$: Asymptotic upper bound (worst-case performance ceiling).
  - $\\Omega(g(n))$: Asymptotic lower bound (best-case performance floor).
  - $\\Theta(g(n))$: Asymptotically tight bound (both upper and lower).
- **Linear Search vs. Binary Search**:
  - Linear Search: $O(n)$ comparisons, works on unsorted collections.
  - Binary Search: $O(\\log n)$ comparisons, strictly requires a sorted sequence.

## 3. Definitions
- **Algorithm**: A sequence of unambiguous computational instructions taking input and producing output.
- **Recurrence Relation**: An equation that recursively defines a function $T(n)$ in terms of smaller input values.
- **Divide and Conquer**: A design paradigm dividing a problem into independent subproblems, solving each recursively, and combining results.

## 4. Important Formulas
- **Binary Search Recurrence**:
  $$T(n) = T(n / 2) + O(1)$$
- **Master Theorem Solution**:
  $$T(n) = O(\\log n)$$
- **Big-O Formal Definition**:
  $$f(n) \\le c \\cdot g(n) \\quad \\text{for all } n \\ge n_0, \\text{ where } c > 0, n_0 > 0$$

## 5. Examples & Case Studies
- **Array of 1 Billion Elements**:
  - Linear Search: Up to $10^9$ operations.
  - Binary Search: At most $\\lceil \\log_2(10^9) \\rceil \\approx 30$ comparisons.
- **Pre-sorting Trade-off**: Sorting an unsorted array takes $O(n \\log n)$; binary search is justified when multiple queries are amortised over the sorted array.

## 6. Final Revision Checklist
- [ ] State the definitions of $O$, $\\Omega$, and $\\Theta$ notation.
- [ ] Write and solve the binary search recurrence relation $T(n) = T(n/2) + O(1)$.
- [ ] Compare linear search $O(n)$ and binary search $O(\\log n)$ on large datasets.
- [ ] Explain why binary search requires random access and pre-sorted inputs.
"""

SAMPLE_FLASHCARDS: list[dict[str, Any]] = [
    {
        "front": "What is the worst-case time complexity of binary search on a sorted array of size n?",
        "back": "O(log n), because the search space is halved at each comparison step.",
        "category": "Formula",
        "timestamp": "01:26",
        "tags": "PrepGenius",
    },
    {
        "front": "What is the recurrence relation that models binary search?",
        "back": "T(n) = T(n/2) + O(1), where O(1) is the constant time to compare the middle element.",
        "category": "Formula",
        "timestamp": "01:14",
        "tags": "PrepGenius",
    },
    {
        "front": "What prerequisite must be satisfied before binary search can be applied?",
        "back": "The input array must be sorted and support constant-time random access.",
        "category": "Concept",
        "timestamp": "01:45",
        "tags": "PrepGenius",
    },
    {
        "front": "Define Big-O asymptotic notation formally.",
        "back": "f(n) = O(g(n)) if there exist positive constants c and n0 such that 0 <= f(n) <= c * g(n) for all n >= n0.",
        "category": "Definition",
        "timestamp": "00:21",
        "tags": "PrepGenius",
    },
    {
        "front": "How does Big-Theta differ from Big-O notation?",
        "back": "Big-O provides an asymptotic upper bound, whereas Big-Theta provides an asymptotically tight bound (both upper and lower).",
        "category": "Relationship",
        "timestamp": "00:27",
        "tags": "PrepGenius",
    },
    {
        "front": "How many comparisons does binary search require in the worst case for one billion elements?",
        "back": "Approximately 30 comparisons, because ceil(log2(10^9)) = 30.",
        "category": "Concept",
        "timestamp": "01:33",
        "tags": "PrepGenius",
    },
    {
        "front": "When is it computationally worthwhile to sort an array before searching?",
        "back": "When multiple searches are performed, amortising the initial O(n log n) sorting cost across queries.",
        "category": "Relationship",
        "timestamp": "01:51",
        "tags": "PrepGenius",
    },
    {
        "front": "What is the worst-case time complexity of linear search?",
        "back": "Theta(n), because every element may need to be inspected sequentially.",
        "category": "Formula",
        "timestamp": "00:45",
        "tags": "PrepGenius",
    },
]

SAMPLE_QUIZ: list[dict[str, Any]] = [
    {
        "question": "What is the worst-case time complexity of binary search on a sorted array of length n?",
        "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
        "answer": "O(log n)",
        "explanation": "At each comparison, binary search discards half of the remaining elements, yielding logarithmic depth.",
        "difficulty": "easy",
    },
    {
        "question": "Which recurrence relation describes the runtime T(n) of binary search?",
        "options": [
            "T(n) = T(n/2) + O(1)",
            "T(n) = 2T(n/2) + O(n)",
            "T(n) = T(n - 1) + O(1)",
            "T(n) = 2T(n/2) + O(1)",
        ],
        "answer": "T(n) = T(n/2) + O(1)",
        "explanation": "Binary search makes one comparison (constant work) and continues in one subproblem of half the size.",
        "difficulty": "medium",
    },
    {
        "question": "What fundamental condition must hold for binary search to work correctly?",
        "options": [
            "The elements must be integers",
            "The array must be sorted and support direct indexing",
            "The array length must be an exact power of two",
            "All elements must be unique",
        ],
        "answer": "The array must be sorted and support direct indexing",
        "explanation": "Binary search requires knowing that elements before mid are smaller and after mid are larger, plus O(1) random access.",
        "difficulty": "easy",
    },
    {
        "question": "Approximately how many comparisons does binary search take on an array of 1,000,000,000 sorted elements?",
        "options": ["30", "1,000", "500,000", "1,000,000"],
        "answer": "30",
        "explanation": "log2(10^9) is approximately 29.897, so at most 30 comparisons are required.",
        "difficulty": "medium",
    },
    {
        "question": "If an array is initially unsorted and you need to perform k queries, when does pre-sorting followed by binary search beat linear search?",
        "options": [
            "When k is significantly greater than log n",
            "When k is equal to 1",
            "Only when n is smaller than 10",
            "Never, because sorting always dominates total runtime",
        ],
        "answer": "When k is significantly greater than log n",
        "explanation": "Sorting takes O(n log n) and k searches take O(k log n). Linear search takes O(k * n). For large k, O(n log n + k log n) < O(k * n).",
        "difficulty": "hard",
    },
    {
        "question": "Which asymptotic notation indicates that an algorithm's growth rate is bounded both above and below by g(n)?",
        "options": ["Big-O: O(g(n))", "Big-Omega: Ω(g(n))", "Big-Theta: Θ(g(n))", "Little-o: o(g(n))"],
        "answer": "Big-Theta: Θ(g(n))",
        "explanation": "Big-Theta defines an asymptotically tight bound, sandwiching the function between c1*g(n) and c2*g(n).",
        "difficulty": "medium",
    },
    {
        "question": "What is the space complexity of standard iterative binary search?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "answer": "O(1)",
        "explanation": "Iterative binary search uses only a few index pointers (low, high, mid) and requires no recursive call stack.",
        "difficulty": "hard",
    },
    {
        "question": "What is the worst-case number of comparisons for linear search on an array of size n?",
        "options": ["n", "log2(n)", "n / 2", "1"],
        "answer": "n",
        "explanation": "If the element is at the very last position or not present at all, linear search checks all n elements.",
        "difficulty": "easy",
    },
]


def get_demo_package(num_cards: int = 15, num_quiz: int = 8) -> dict[str, Any]:
    """
    Return a complete demo bundle with sample transcript, notes, flashcards,
    and quiz for offline or API-key-free usage.
    """
    return {
        "title": SAMPLE_LECTURE_TITLE,
        "segments": SAMPLE_TRANSCRIPT_SEGMENTS,
        "raw_transcript": " ".join(s["text"] for s in SAMPLE_TRANSCRIPT_SEGMENTS),
        "notes": SAMPLE_REVISION_NOTES,
        "flashcards": SAMPLE_FLASHCARDS[:num_cards],
        "quiz": SAMPLE_QUIZ[:num_quiz],
    }
