/**
 * F-score contour math utilities
 *
 * F = 2PR / (P + R)
 * Solving for P: P = (F * R) / (2R - F)
 * Valid when R > F/2
 */

export interface Point {
  x: number  // recall
  y: number  // precision
}

/**
 * Generate points for an iso-F contour line
 * @param fScore The F-score value for this contour (e.g., 0.1, 0.2, ..., 0.9)
 * @param numPoints Number of points to generate
 * @returns Array of {x: recall, y: precision} points
 */
export function generateFscoreContour(fScore: number, numPoints = 500): Point[] {
  const points: Point[] = []

  // F = 2PR/(P+R), solving for P given R:
  // P = FR/(2R-F)
  // Valid only when 2R > F, i.e., R > F/2

  const minRecall = fScore / 2 + 0.001 // slightly above F/2 to avoid division issues

  for (let i = 0; i <= numPoints; i++) {
    const recall = minRecall + (1 - minRecall) * (i / numPoints)

    if (recall <= fScore / 2) continue

    const precision = (fScore * recall) / (2 * recall - fScore)

    // Only include points within valid range
    if (precision >= 0 && precision <= 1) {
      points.push({ x: recall, y: precision })
    }
  }

  return points
}

/**
 * Generate multiple F-score contour lines
 * @param fScores Array of F-score values (e.g., [0.1, 0.2, 0.3, ...])
 * @param numPointsPerContour Points per contour line
 * @returns Map of F-score to points array
 */
export function generateAllContours(
  fScores: number[] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  numPointsPerContour = 200
): Map<number, Point[]> {
  const contours = new Map<number, Point[]>()

  for (const fScore of fScores) {
    contours.set(fScore, generateFscoreContour(fScore, numPointsPerContour))
  }

  return contours
}

/**
 * Calculate F-score from precision and recall
 */
export function calculateFscore(precision: number, recall: number): number {
  if (precision + recall === 0) return 0
  return (2 * precision * recall) / (precision + recall)
}

/**
 * Get label position for an F-score contour
 * @param fScore The F-score value
 * @returns Position {x, y} for the label
 */
export function getContourLabelPosition(fScore: number): Point {
  // Place label at the point where recall equals the F-score
  // This gives a nice diagonal arrangement
  const recall = Math.max(fScore, fScore / 2 + 0.1)
  if (recall > fScore / 2) {
    const precision = (fScore * recall) / (2 * recall - fScore)
    if (precision >= 0 && precision <= 1) {
      return { x: recall, y: precision }
    }
  }
  // Fallback
  return { x: fScore, y: fScore }
}
