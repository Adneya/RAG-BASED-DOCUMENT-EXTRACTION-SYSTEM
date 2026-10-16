export type RetrievedContext = {
	chunk_id: string
	document_id: string
	filename: string
	source_type: string
	text: string
	score: number
	chunk_index: number
	page_number?: number | null
}

export type RagDocument = {
	document_id: string
	filename: string
	source_type: string
	chunk_count: number
	token_count: number
}

export type PlanStep = {
	tool: string
	action: string
	status: string
	detail: string
}

export type AgentResponse = {
	status: string
	final_output: string
	retrieved_contexts: RetrievedContext[]
	plan_trace: PlanStep[]
	cost_estimate: {
		estimated_input_tokens: number
		estimated_output_tokens: number
		estimated_usd: number | null
		note: string
	}
	perspectives: ResearchPerspective[]
}

export type ResearchPerspective = { key: 'executive' | 'evidence' | 'risks' | 'opportunities' | 'actions'; title: string; summary: string; confidence: number; sources: string[] }

export type AnalyticsColumn = { name: string; kind: 'date' | 'number' | 'category' | 'text'; non_empty: number; unique_values: number }
export type AnalyticsPoint = { label: string; value: number }
export type AnalyticsInsight = { finding: string; impact: string; action: string; confidence: number; source: string }
export type AnalyticsResponse = {
	status: string
	filename: string
	row_count: number
	columns: AnalyticsColumn[]
	date_column: string | null
	measure_column: string | null
	category_column: string | null
	kpis: Record<string, number>
	trend: AnalyticsPoint[]
	categories: AnalyticsPoint[]
	anomalies: AnalyticsPoint[]
	distribution: AnalyticsPoint[]
	insights: AnalyticsInsight[]
	warnings: string[]
}

export type AuthUser = { id: string; name: string; email: string }
