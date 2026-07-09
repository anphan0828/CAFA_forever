export interface Point {
  x: number
  y: number
}

export function generateFScoreContour(fScore: number, steps = 100): Point[] {
  const points: Point[] = []

  for (let index = 1; index <= steps; index += 1) {
    const recall = index / steps
    if (recall <= fScore / 2) continue

    const precision = (fScore * recall) / (2 * recall - fScore)
    if (precision >= 0 && precision <= 1) {
      points.push({ x: recall, y: precision })
    }
  }

  return points
}

export function generateAllContours(fScores: number[], steps = 100): Map<number, Point[]> {
  return new Map(fScores.map((fScore) => [fScore, generateFScoreContour(fScore, steps)]))
}
