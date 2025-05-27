// Article utilities for formatting and calculations

/**
 * Formats a date string into a human-readable label (e.g., Today, Yesterday, X days ago, or short date).
 */
export function formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    if (diffDays <= 1) return 'Today';
    if (diffDays <= 2) return 'Yesterday';
    if (diffDays <= 7) return `${diffDays} days ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/**
 * Estimates reading time in minutes based on description length, or uses provided value if available.
 */
export function getReadingTime(description: string, readingTime?: number): number {
    if (readingTime) return readingTime;
    const wordCount = description.split(/\s+/).length;
    return Math.max(1, Math.round(wordCount / 200));
} 