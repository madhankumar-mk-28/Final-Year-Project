/**
 * api.js
 * -------
 * All fetch calls to the Flask backend (http://localhost:5000).
 * Import these helpers from any React component.
 *
 * Endpoints covered:
 *   uploadResumes(files)             → POST /api/upload-resumes
 *   runScreening(payload)            → POST /api/screen
 *   getResults()                     → GET  /api/results
 *   getCandidate(id)                 → GET  /api/candidate/:id
 *   getConfig()                      → GET  /api/config
 *   saveConfig(cfg)                  → POST /api/config
 *   getActiveModel()                 → GET  /api/model
 *   switchModel(modelKey)            → POST /api/model
 *   clearUploads()                   → POST /api/clear
 */

const BASE = "http://localhost:5000";

// ─── Generic fetch wrapper ────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
    const res = await fetch(`${BASE}${path}`, options);
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
}

// ─── Resume upload ────────────────────────────────────────────────────────────

/**
 * Upload PDF resumes.
 * @param {FileList|File[]} files
 * @returns {{ message, saved, rejected }}
 */
export async function uploadResumes(files) {
    const formData = new FormData();
    Array.from(files).forEach(f => formData.append("files", f));
    return apiFetch("/api/upload-resumes", {
        method: "POST",
        body: formData,
    });
}

// ─── Screening pipeline ───────────────────────────────────────────────────────

/**
 * Trigger the full ML screening pipeline.
 * @param {{
 *   job_description: string,
 *   required_skills: string[],
 *   model?: "mpnet" | "minilm",
 *   config?: {
 *     min_experience_years?: number,
 *     skill_weight?: number,
 *     semantic_weight?: number,
 *     top_n?: number,
 *   }
 * }} payload
 * @returns {{ message, model_used, total, shortlisted, results }}
 */
export async function runScreening(payload) {
    return apiFetch("/api/screen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

// ─── Results ──────────────────────────────────────────────────────────────────

/**
 * Fetch the most recent screening results.
 * @returns {{ results, total }}
 */
export async function getResults() {
    return apiFetch("/api/results");
}

/**
 * Fetch full detail for a single candidate.
 * @param {number} id
 */
export async function getCandidate(id) {
    return apiFetch(`/api/candidate/${id}`);
}

// ─── Config ───────────────────────────────────────────────────────────────────

/**
 * Fetch the current config.json contents.
 */
export async function getConfig() {
    return apiFetch("/api/config");
}

/**
 * Save config to config.json.
 * @param {{ job_description, required_skills, model, scoring }} cfg
 */
export async function saveConfig(cfg) {
    return apiFetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
    });
}

// ─── Model management ─────────────────────────────────────────────────────────

/**
 * Get the currently active model.
 * @returns {{ active, full_name, available }}
 */
export async function getActiveModel() {
    return apiFetch("/api/model");
}

/**
 * Switch the active model.
 * @param {"mpnet" | "minilm"} modelKey
 * @returns {{ message, active, full_name }}
 */
export async function switchModel(modelKey) {
    return apiFetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey }),
    });
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/**
 * Clear all uploaded PDFs and results from the server.
 */
export async function clearUploads() {
    return apiFetch("/api/clear", { method: "POST" });
}