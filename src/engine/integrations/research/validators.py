"""
Validation utilities for research quality and credibility
"""

import re
from typing import Dict, List, Any, Set
import logging


class SourceValidator:
    """Validate source credibility and calculate quality metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Domain credibility mappings
        self.high_credibility_domains = {
            "reuters.com",
            "bbc.com",
            "apnews.com",
            "npr.org",
            "pbs.org",
            "nature.com",
            "science.org",
            "cell.com",
            "nejm.org",
            "bmj.com",
            "nih.gov",
            "cdc.gov",
            "who.int",
            "fda.gov",
            "sec.gov",
            "harvard.edu",
            "mit.edu",
            "stanford.edu",
            "cambridge.org",
            "jstor.org",
            "worldbank.org",
            "imf.org",
            "oecd.org",
            "un.org",
            "economist.com",
        }

        self.medium_credibility_domains = {
            "wsj.com",
            "ft.com",
            "bloomberg.com",
            "fortune.com",
            "forbes.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "cnn.com",
            "techcrunch.com",
            "wired.com",
            "arstechnica.com",
            "spectrum.ieee.org",
        }

    def extract_domain(self, url: str) -> str:
        """Extract eTLD+1 domain from URL"""
        try:
            # Simple domain extraction (eTLD+1 approximation)
            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if domain_match:
                domain = domain_match.group(1)
                # Remove common subdomains for better grouping
                domain = re.sub(r"^(www|m|mobile)\.", "", domain)
                return domain
            return "unknown"
        except Exception:
            return "unknown"

    def calculate_source_credibility(self, source: Dict[str, Any]) -> float:
        """Calculate credibility score for a source based on multiple factors"""
        url = source.get("url", "")
        title = source.get("title", "")
        content = source.get("content", "")
        domain = self.extract_domain(url)

        credibility_score = 0.5  # Base score

        # Domain credibility
        if domain in self.high_credibility_domains:
            credibility_score += 0.4
        elif domain in self.medium_credibility_domains:
            credibility_score += 0.2
        elif domain.endswith(".edu") or domain.endswith(".gov"):
            credibility_score += 0.3
        elif domain.endswith(".org"):
            credibility_score += 0.1

        # Content quality indicators
        text_length = len(content)
        if text_length > 500:  # Substantial content
            credibility_score += 0.1

        # Check for citations and references
        citation_patterns = [
            r"according to",
            r"study by",
            r"research from",
            r"data from",
            r"survey by",
            r"report by",
            r"analysis by",
            r"findings from",
        ]

        citation_count = 0
        content_lower = content.lower()
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content_lower))

        if citation_count >= 3:
            credibility_score += 0.1
        elif citation_count >= 1:
            credibility_score += 0.05

        # Check for numeric data (suggests factual content)
        numeric_data = re.findall(r"\d+(?:\.\d+)?%?", content)
        if len(numeric_data) >= 5:
            credibility_score += 0.1
        elif len(numeric_data) >= 2:
            credibility_score += 0.05

        # Penalize sensational language
        sensational_words = [
            "shocking",
            "amazing",
            "incredible",
            "unbelievable",
            "secret",
            "you won't believe",
            "doctors hate",
            "one weird trick",
        ]

        title_content_lower = f"{title} {content}".lower()
        sensational_count = sum(
            1 for word in sensational_words if word in title_content_lower
        )

        if sensational_count > 0:
            credibility_score -= 0.1 * sensational_count

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, credibility_score))


class ConsistencyValidator:
    """Validate consistency across sources and detect contradictions"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_coverage_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate coverage score based on domain diversity"""
        if not sources:
            return 0.0

        validator = SourceValidator()
        unique_domains = set()
        for source in sources:
            url = source.get("url", "")
            domain = validator.extract_domain(url)
            if domain != "unknown":
                unique_domains.add(domain)

        coverage_score = min(1.0, len(unique_domains) / 5.0)
        return coverage_score

    def calculate_title_similarity(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate average Jaccard similarity of titles"""
        if len(sources) < 2:
            return 1.0  # Single source = perfect consistency

        def normalize_title(title: str) -> Set[str]:
            """Normalize title to word set"""
            # Remove stopwords and normalize
            stopwords = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
            }
            words = re.findall(r"\w+", title.lower())
            return set(
                word for word in words if len(word) > 2 and word not in stopwords
            )

        titles = [source.get("title", "") for source in sources]
        normalized_titles = [normalize_title(title) for title in titles if title]

        if len(normalized_titles) < 2:
            return 0.5  # Insufficient data

        similarities = []
        for i in range(len(normalized_titles)):
            for j in range(i + 1, len(normalized_titles)):
                set1, set2 = normalized_titles[i], normalized_titles[j]
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
                similarities.append(jaccard)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def detect_numeric_agreement(self, sources: List[Dict[str, Any]]) -> float:
        """Detect numeric claims and check agreement within tolerance"""
        # Simple numeric extraction from titles/content
        numeric_pattern = r"(\d+(?:\.\d+)?)\s*%?"

        all_numbers = []
        for source in sources:
            text = f"{source.get('title', '')} {source.get('content', '')}"
            numbers = [float(match) for match in re.findall(numeric_pattern, text)]
            all_numbers.extend(numbers)

        if len(all_numbers) < 2:
            return 1.0  # No contradictions found

        # Check if numbers are within ±10% tolerance
        agreements = 0
        total_pairs = 0

        for i in range(len(all_numbers)):
            for j in range(i + 1, len(all_numbers)):
                num1, num2 = all_numbers[i], all_numbers[j]
                total_pairs += 1

                # Calculate relative tolerance
                if abs(num1) < 1.0 and abs(num2) < 1.0:
                    # Absolute tolerance for small numbers
                    tolerance = 0.1
                    if abs(num1 - num2) <= tolerance:
                        agreements += 1
                else:
                    # Relative tolerance for larger numbers
                    avg = (abs(num1) + abs(num2)) / 2
                    if avg > 0 and abs(num1 - num2) / avg <= 0.1:
                        agreements += 1

        return agreements / total_pairs if total_pairs > 0 else 1.0

    def calculate_consistency_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate consistency score using multiple heuristics"""
        if not sources:
            return 0.0

        title_similarity = self.calculate_title_similarity(sources)
        numeric_agreement = self.detect_numeric_agreement(sources)
        stance_alignment = 1.0  # Simplified - assume no contradictory stance

        consistency_score = (
            0.5 * title_similarity + 0.4 * numeric_agreement + 0.1 * stance_alignment
        )

        return min(1.0, consistency_score)

    def detect_contradictions(
        self, sources: List[Dict[str, Any]], facts: List[str]
    ) -> List[str]:
        """Detect contradictory claims across sources"""
        contradictions = []

        # Extract numeric claims for contradiction detection
        numeric_claims = {}  # topic -> [values]

        for source in sources:
            text = f"{source.get('title', '')} {source.get('content', '')}"

            # Look for percentage claims
            percentage_matches = re.finditer(
                r"(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(\d+(?:\.\d+)?)\s*%",
                text,
                re.IGNORECASE,
            )
            for match in percentage_matches:
                topic = match.group(1).lower().strip()
                value = float(match.group(2))

                if topic not in numeric_claims:
                    numeric_claims[topic] = []
                numeric_claims[topic].append(value)

        # Check for contradictory numeric claims
        for topic, values in numeric_claims.items():
            if len(values) >= 2:
                min_val = min(values)
                max_val = max(values)

                # If values differ by more than 20%, it's a contradiction
                if max_val > 0 and (max_val - min_val) / max_val > 0.2:
                    contradictions.append(
                        f"Contradictory data for {topic}: {min_val}% vs {max_val}%"
                    )

        return contradictions

    def calculate_cross_reference_score(
        self, sources: List[Dict[str, Any]], facts: List[str]
    ) -> float:
        """Calculate how well sources cross-reference each other"""
        if len(sources) < 2:
            return 0.0

        validator = SourceValidator()
        cross_references = 0
        total_possible = 0

        # Check if sources reference each other's domains
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if i >= j:
                    continue

                total_possible += 1

                domain1 = validator.extract_domain(source1.get("url", ""))
                domain2 = validator.extract_domain(source2.get("url", ""))

                # Check if source1 mentions domain2 or vice versa
                content1 = source1.get("content", "").lower()
                content2 = source2.get("content", "").lower()

                if domain2 in content1 or domain1 in content2:
                    cross_references += 1
                    continue

                # Check for shared factual claims
                shared_facts = 0
                for fact in facts:
                    fact_words = set(re.findall(r"\w+", fact.lower()))
                    content1_words = set(re.findall(r"\w+", content1))
                    content2_words = set(re.findall(r"\w+", content2))

                    overlap1 = (
                        len(fact_words & content1_words) / len(fact_words)
                        if fact_words
                        else 0
                    )
                    overlap2 = (
                        len(fact_words & content2_words) / len(fact_words)
                        if fact_words
                        else 0
                    )

                    if (
                        overlap1 > 0.3 and overlap2 > 0.3
                    ):  # Both sources support this fact
                        shared_facts += 1

                if shared_facts >= 2:  # At least 2 shared facts
                    cross_references += 1

        return cross_references / total_possible if total_possible > 0 else 0.0
