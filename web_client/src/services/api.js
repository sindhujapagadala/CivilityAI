/**
 * CivilityAI API Service Integration.
 * Dispatches requests to the FastAPI backend service.
 */

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = 'API request failed';
    try {
      const err = await response.json();
      errorDetail = err.detail || err.message || errorDetail;
    } catch {
      errorDetail = response.statusText;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const apiService = {
  /**
   * System health status.
   */
  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/system/health`);
    return handleResponse(response);
  },

  /**
   * Submit single message for multi-label safety assessment.
   */
  async analyzeMessage(messageBody) {
    const response = await fetch(`${API_BASE_URL}/safety/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message_body: messageBody }),
    });
    return handleResponse(response);
  },

  /**
   * Fetch queue of flagged messages needing human moderation review.
   */
  async getPendingReviews(limit = 50, offset = 0) {
    const response = await fetch(`${API_BASE_URL}/review/pending?limit=${limit}&offset=${offset}`);
    return handleResponse(response);
  },

  /**
   * Submit human moderator resolution action.
   */
  async submitReviewAction(recordId, reviewAction, reviewerIdentifier = 'console_moderator', notes = '') {
    const response = await fetch(`${API_BASE_URL}/review/${recordId}/action`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        review_action: reviewAction,
        reviewer_identifier: reviewerIdentifier,
        notes: notes,
      }),
    });
    return handleResponse(response);
  },

  /**
   * Retrieve platform analytics and telemetry.
   */
  async getAnalyticsSummary() {
    const response = await fetch(`${API_BASE_URL}/analytics/summary`);
    return handleResponse(response);
  },

  /**
   * Upload CSV for batch analysis.
   */
  async uploadBatchCsv(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/batch/analyze`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },
};
