export type ClassCount = '2' | '3'
export interface ModelRun {
  run: string
  classes: number
  mode: string
  fusion: string
  metrics: Record<string, unknown>
}
export interface ModelsResponse {
  runs: ModelRun[]
  defaults: Record<ClassCount, string>
}
export interface PredictionResponse {
  run: string
  class_names: string[]
  probs: Record<string, number>
  pred: string
  confidence: number
  gradcam: string
  lime: string
}
export interface ShapResponse { shap: string }
// CSV values may arrive as strings; empty metrics are also valid.
export interface AblationRow {
  run: string
  params_M: number | string | null
  accuracy: number | string | null
  precision: number | string | null
  recall: number | string | null
  f1: number | string | null
  roc_auc: number | string | null
}
export interface RunFigures {
  confusion?: string | null
  roc?: string | null
  curves?: string | null
}
export interface ResultsResponse {
  ablation: AblationRow[]
  figures: Record<string, RunFigures>
}
export type SamplesResponse = string[]
