import type { AgentResponse, AnalyticsResponse, AuthUser, RagDocument } from './types'

const request = async <T>(input: RequestInfo, init?: RequestInit): Promise<T> => {
	const response = await fetch(input, init)
	if (!response.ok) throw new Error(await response.text() || `Request failed: ${response.status}`)
	return response.json() as Promise<T>
}

export const fetchCurrentUser = () => request<AuthUser>('/api/auth/me')
export const registerAccount = (name: string, email: string, password: string) => request<AuthUser>('/api/auth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, email, password }) })
export const loginAccount = (email: string, password: string) => request<AuthUser>('/api/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) })
export const logoutAccount = () => request<{ status: string }>('/api/auth/logout', { method: 'POST' })

export const fetchDocuments = () => request<RagDocument[]>('/api/rag/documents')

export const askQuestion = (query: string, files: File[]) => {
	const body = new FormData()
	body.append('query', query)
	files.forEach((file) => body.append('files', file))
	return request<AgentResponse>('/api/chat', { method: 'POST', body })
}

export const analyzeTable = (file: File, fields?: { dateColumn?: string; measureColumn?: string; categoryColumn?: string }) => {
	const body = new FormData()
	body.append('file', file)
	if (fields?.dateColumn) body.append('date_column', fields.dateColumn)
	if (fields?.measureColumn) body.append('measure_column', fields.measureColumn)
	if (fields?.categoryColumn) body.append('category_column', fields.categoryColumn)
	return request<AnalyticsResponse>('/api/analytics/analyze', { method: 'POST', body })
}

export const deleteDocument = (documentId: string) =>
	request(`/api/rag/documents/${documentId}`, { method: 'DELETE' })
