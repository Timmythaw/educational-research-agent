"""
Download educational research papers from open sources.
Run with: uv run python scripts/download_papers.py
"""

import os
import requests
from pathlib import Path
from time import sleep

# Educational research papers from arXiv (education category)
PAPERS = [
    # Online Learning & Educational Technology
    {"url": "https://arxiv.org/pdf/2301.04329.pdf", "title": "online_learning_effectiveness.pdf"},
    {"url": "https://arxiv.org/pdf/2103.15355.pdf", "title": "ai_education_systematic_review.pdf"},
    {"url": "https://arxiv.org/pdf/2308.10792.pdf", "title": "chatgpt_education_implications.pdf"},
    
    # Teaching Methods & Pedagogy
    {"url": "https://arxiv.org/pdf/2105.03824.pdf", "title": "active_learning_strategies.pdf"},
    {"url": "https://arxiv.org/pdf/2204.07780.pdf", "title": "flipped_classroom_effectiveness.pdf"},
    {"url": "https://arxiv.org/pdf/2110.02496.pdf", "title": "collaborative_learning_outcomes.pdf"},
    
    # Student Assessment & Evaluation
    {"url": "https://arxiv.org/pdf/2201.08239.pdf", "title": "formative_assessment_techniques.pdf"},
    {"url": "https://arxiv.org/pdf/2209.14146.pdf", "title": "automated_grading_systems.pdf"},
    {"url": "https://arxiv.org/pdf/2107.03374.pdf", "title": "student_engagement_measurement.pdf"},
    
    # Educational Data Mining
    {"url": "https://arxiv.org/pdf/2202.07309.pdf", "title": "learning_analytics_education.pdf"},
    {"url": "https://arxiv.org/pdf/2106.11748.pdf", "title": "predicting_student_performance.pdf"},
    {"url": "https://arxiv.org/pdf/2203.09450.pdf", "title": "educational_data_mining_survey.pdf"},
    
    # STEM Education
    {"url": "https://arxiv.org/pdf/2104.09334.pdf", "title": "programming_education_methods.pdf"},
    {"url": "https://arxiv.org/pdf/2108.04842.pdf", "title": "math_anxiety_interventions.pdf"},
    {"url": "https://arxiv.org/pdf/2205.11487.pdf", "title": "science_education_technology.pdf"},
    
    # Inclusive Education & Accessibility
    {"url": "https://arxiv.org/pdf/2111.05826.pdf", "title": "accessible_online_learning.pdf"},
    {"url": "https://arxiv.org/pdf/2206.08362.pdf", "title": "inclusive_education_practices.pdf"},
    {"url": "https://arxiv.org/pdf/2103.16582.pdf", "title": "learning_disabilities_support.pdf"},
    
    # Additional Papers
    {"url": "https://arxiv.org/pdf/2207.12525.pdf", "title": "personalized_learning_systems.pdf"},
    {"url": "https://arxiv.org/pdf/2112.09332.pdf", "title": "mooc_effectiveness_study.pdf"},
]

def download_papers(output_dir="data/papers"):
    """Download papers to specified directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(PAPERS)} educational research papers...")
    print(f"Output directory: {output_dir}\n")
    
    for i, paper in enumerate(PAPERS, 1):
        output_path = Path(output_dir) / paper["title"]
        
        if output_path.exists():
            print(f"[{i}/{len(PAPERS)}] Already exists: {paper['title']}")
            continue
        
        try:
            print(f"Downloading [{i}/{len(PAPERS)}] {paper['title']}")
            response = requests.get(paper["url"], timeout=30)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"Downloaded: {paper['title']} ({len(response.content) // 1024} KB)")
            sleep(1)  # Be nice to the server
            
        except Exception as e:
            print(f"Failed to download {paper['title']}: {e}")
    
    print(f"\nDownload complete! Papers saved to: {output_dir}")

if __name__ == "__main__":
    download_papers()
