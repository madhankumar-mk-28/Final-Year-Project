import { useState, useRef, useCallback, useEffect } from "react";
import {
    LayoutDashboard, Briefcase, Users, BarChart2, Upload, X, Check, Plus,
    FileText, TrendingUp, Award, Mail, ChevronRight,
    Sparkles, Clock, Star, AlertCircle, CheckCircle, Zap, ArrowRight,
    RefreshCw, Sun, Moon, Edit3, Save, Activity, Target, Download,
    Menu, ChevronDown, Info, Phone,
} from "lucide-react";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, Radar, CartesianGrid,
} from "recharts";

// ─── THEMES ───────────────────────────────────────────────────────────────────
// Palette: Tailwind CSS v3 tokens (industry standard — used by Vercel, Linear, etc.)
const DARK = {
    // Zinc-950 base — the same near-black Vercel uses
    bg: "#09090b", surface: "rgba(255,255,255,0.04)", border: "rgba(255,255,255,0.09)",
    text: "#f4f4f5", sub: "#71717a", muted: "#52525b",
    // Indigo-500 / Green-500 / Amber-500 / Pink-500 / Teal-500
    blue: "#6263c6", green: "#22c55e", amber: "#f59e0b", pink: "#ec4899", teal: "#14b8a6",
    sidebarBg: "rgba(255,255,255,0.02)", topbarBg: "rgba(9,9,11,0.90)",
    inputBg: "rgba(255,255,255,0.05)", cardText: "#d4d4d8",
    drawerBg: "#0f0f11", scrollThumb: "rgba(255,255,255,0.09)",
    chartTooltip: "#0f0f11", tableFocus: "rgba(99,102,241,0.06)",
    // Chart: Indigo-400 · Emerald-400 · Amber-400 — vibrant but not neon
    chartSkill: "#818cf8", chartSemantic: "#34d399", chartFinal: "#fbbf24",
};
const LIGHT = {
    // Dimmer blue-grey base — noticeably muted, not blinding
    bg: "#c8cdd5", surface: "#d2d7df", border: "rgba(0,0,0,0.11)",
    // Softer text — readable but not pure black
    text: "#1e2533", sub: "#374151", muted: "#5c6878",
    // Mid-tone accents — visible on the dimmer bg without being harsh
    blue: "#4338ca", green: "#166534", amber: "#a16207", pink: "#be185d", teal: "#0f766e",
    sidebarBg: "#bcc2cb", topbarBg: "rgba(200,205,213,0.95)",
    inputBg: "rgba(0,0,0,0.06)", cardText: "#2d3748",
    drawerBg: "#c4c9d1", scrollThumb: "rgba(0,0,0,0.14)",
    chartTooltip: "#e2e6ea", tableFocus: "rgba(67,56,202,0.06)",
    chartSkill: "#4338ca", chartSemantic: "#166534", chartFinal: "#a16207",
};

// Helper: derive a short display name that avoids truncating on a 1-2 char prefix
const shortDisplayName = (fullName, maxLen = 13) => {
    const parts = (fullName || "?").split(" ").filter(Boolean);
    let nm = parts[0] || "?";
    let pi = 1;
    while (nm.length <= 2 && pi < parts.length) { nm = parts.slice(0, pi + 1).join(" "); pi++; }
    return nm.slice(0, maxLen);
};

// Module-level theme ref — updated at top of each App render
let C = DARK;

// Helper: base card style with optional overrides
const card = (extra = {}) => ({
    background: C.surface, border: `1px solid ${C.border}`, borderRadius: 16, padding: 24, ...extra,
});

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const BASE = "http://localhost:5001";   // Flask backend URL

const PIPELINE_STEPS = (skillW = 55, modelName = "MPNet") => [
    { label: "resume_parser.py", desc: "Extracting text from PDFs via pdfplumber" },
    { label: "information_extractor.py", desc: "NER + regex — names, email, phone, skills, experience" },
    { label: "semantic_matcher.py", desc: `Generating sentence embeddings using ${modelName}` },
    { label: "semantic_matcher.py", desc: "Computing cosine similarity vs. job description" },
    { label: "scoring_engine.py", desc: "Applying eligibility filter (min experience check)" },
    { label: "scoring_engine.py", desc: `Weighted ranking — ${skillW}% skill / ${100 - skillW}% semantic` },
];

// Three supported embedding models — matches VALID_MODELS in app.py
const MODELS = {
    mpnet: {
        key: "mpnet",
        name: "MPNet",
        short: "multi-qa-mpnet-base-dot-v1",
        badge: "Best Accuracy",
        color: "#818cf8",
        desc: "Sentence-BERT (SBERT) model trained on 215M question-answer pairs using a Siamese network — the only true SBERT model in this system and the most semantically aware for resume matching.",
        detail: "768-dim · SBERT architecture · dot-product similarity · best for demos & high accuracy.",
    },
    mxbai: {
        key: "mxbai",
        name: "MxBai",
        short: "mixedbread-ai/mxbai-embed-large-v1",
        badge: "2025 MTEB #1",
        color: "#e879f9",
        desc: "State-of-the-art embedding model by Mixedbread AI — ranked #1 on the MTEB English leaderboard. Uses contrastive training for superior retrieval and ranking performance.",
        detail: "1024-dim · 335M params · contrastive training · highest real-world retrieval accuracy.",
    },
    arctic: {
        key: "arctic",
        name: "Arctic",
        short: "Snowflake/snowflake-arctic-embed-m-v1.5",
        badge: "Top Ranking",
        color: "#38bdf8",
        desc: "Snowflake's enterprise embedding model — not SBERT, but a dedicated retrieval model optimised for high-precision document ranking and search at scale.",
        detail: "768-dim · MTEB top retrieval · enterprise-grade · ideal for high-precision screening.",
    },
};

// ─── HOOKS ────────────────────────────────────────────────────────────────────
function useWindowWidth() {
    const [w, setW] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);
    useEffect(() => {
        const h = () => setW(window.innerWidth);
        window.addEventListener("resize", h);
        return () => window.removeEventListener("resize", h);
    }, []);
    return w;
}

// ─── SHARED PRIMITIVES ────────────────────────────────────────────────────────
const CircularRing = ({ value, size = 52, sw = 4.5, color }) => {
    const clr = color || C.blue;
    const safeVal = Math.min(100, Math.max(0, value || 0));
    const r = (size - sw) / 2;
    const circ = r * 2 * Math.PI;
    const offset = circ - (safeVal / 100) * circ;
    return (
        <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
            <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(128,128,128,0.13)" strokeWidth={sw} />
                <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={clr} strokeWidth={sw}
                    strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }} />
            </svg>
            <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: size * 0.22, fontWeight: 700, color: C.text }}>
                {safeVal}
            </span>
        </div>
    );
};

const GaugeArc = ({ value }) => {
    const r = 64, sw = 12, cx = 85, cy = 82;
    const circ = Math.PI * r;
    const off = circ - (value / 100) * circ;
    const col = value >= 80 ? C.green : value >= 55 ? C.blue : C.amber;
    return (
        <svg width={170} height={90} viewBox="0 0 170 90">
            <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="rgba(128,128,128,0.1)" strokeWidth={sw} strokeLinecap="round" />
            <path d={`M${cx - r} ${cy} A${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={col} strokeWidth={sw} strokeLinecap="round"
                strokeDasharray={circ} strokeDashoffset={off}
                style={{ transition: "stroke-dashoffset 1.2s ease", filter: `drop-shadow(0 0 7px ${col}88)` }} />
            {/* Score centred vertically inside the arc */}
            <text x={cx} y={cy - 16} textAnchor="middle" fill={C.text} fontSize="26" fontWeight="900">{value}</text>
            <text x={cx} y={cy - 2} textAnchor="middle" fill={C.sub} fontSize="10">/ 100</text>
        </svg>
    );
};

const Badge = ({ status }) => {
    const m = {
        Shortlisted: { bg: "rgba(34,197,94,.12)", col: "#22c55e" },
        Rejected: { bg: "rgba(239,68,68,.12)", col: "#ef4444" },
        Reviewing: { bg: "rgba(245,158,11,.12)", col: "#f59e0b" },
    };
    const s = m[status] || m.Reviewing;
    return (
        <span style={{ padding: "3px 10px", borderRadius: 20, fontSize: 11, fontWeight: 600, background: s.bg, color: s.col, border: `1px solid ${s.col}40`, display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 5, height: 5, borderRadius: "50%", background: s.col }} />{status}
        </span>
    );
};

const SkillChip = ({ label, matched }) => (
    <span style={{
        padding: "3px 10px", borderRadius: 6, fontSize: 11, fontWeight: 500,
        background: matched ? `${C.blue}22` : C.inputBg,
        color: matched ? C.blue : C.sub,
        border: `1px solid ${matched ? `${C.blue}44` : C.border}`,
        display: "inline-flex", alignItems: "center", gap: 4,
    }}>
        {matched ? <Check size={9} /> : <X size={9} />} {label}
    </span>
);

const Toast = ({ msg, type, onClose }) => (
    <div style={{
        position: "fixed", bottom: 24, right: 24, zIndex: 9999,
        padding: "12px 18px", borderRadius: 12,
        background: type === "error" ? "rgba(239,68,68,.15)" : "rgba(34,197,94,.15)",
        border: `1px solid ${type === "error" ? "#ef444460" : "#22c55e60"}`,
        color: type === "error" ? "#ef4444" : C.green,
        display: "flex", alignItems: "center", gap: 10, fontSize: 13, fontWeight: 500,
        boxShadow: "0 8px 32px rgba(0,0,0,.35)", animation: "fadeSlideUp .3s ease", maxWidth: "90vw",
    }}>
        {type === "error" ? <AlertCircle size={15} /> : <CheckCircle size={15} />}
        {msg}
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "inherit", opacity: .6, marginLeft: 4 }}><X size={13} /></button>
    </div>
);

const EmptyState = ({ icon: Icon, title, sub, action }) => (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 24px", textAlign: "center", gap: 12 }}>
        <div style={{ width: 56, height: 56, borderRadius: 16, background: C.inputBg, border: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Icon size={24} color={C.sub} />
        </div>
        <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>{title}</div>
        <div style={{ fontSize: 13, color: C.sub, maxWidth: 280, lineHeight: 1.6 }}>{sub}</div>
        {action}
    </div>
);

// ─── MODEL DROPDOWN ───────────────────────────────────────────────────────────
const ModelDropdown = ({ activeModel, onChange }) => {
    const [open, setOpen] = useState(false);
    const ref = useRef();
    const m = MODELS[activeModel] || MODELS.mpnet;

    // Close dropdown when clicking outside
    useEffect(() => {
        const handler = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
        document.addEventListener("mousedown", handler);
        return () => document.removeEventListener("mousedown", handler);
    }, []);

    return (
        <div ref={ref} style={{ position: "relative" }}>
            {/* Trigger button */}
            <button onClick={() => setOpen(o => !o)} style={{
                display: "flex", alignItems: "center", gap: 7,
                padding: "6px 11px", borderRadius: 10,
                background: `${m.color}14`, border: `1px solid ${m.color}33`,
                cursor: "pointer", fontFamily: "inherit", transition: "background .15s",
            }}>
                {/* Live indicator dot */}
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: m.color, boxShadow: `0 0 6px ${m.color}` }} />
                <span style={{ fontSize: 11, fontWeight: 700, color: m.color }}>{m.name}</span>
                {/* Chevron — vertically centred with translateY so it aligns with text */}
                <ChevronDown size={11} color={m.color} style={{
                    transition: "transform .18s",
                    transform: open ? "rotate(180deg)" : "rotate(0deg)",
                    display: "block",
                    marginTop: 1,   /* fine-tune arrow alignment with text */
                }} />
            </button>

            {/* Dropdown panel */}
            {open && (
                <div style={{
                    position: "absolute", top: "calc(100% + 8px)", right: 0, width: 310,
                    borderRadius: 14, overflow: "hidden",
                    background: C.drawerBg, border: `1px solid ${C.border}`,
                    boxShadow: "0 16px 48px rgba(0,0,0,.35)", zIndex: 200,
                    animation: "fadeSlideUp .15s ease",
                }}>
                    <div style={{ padding: "10px 14px 7px", fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em" }}>
                        Select Embedding Model
                    </div>
                    {Object.values(MODELS).map(opt => {
                        const isActive = activeModel === opt.key;
                        return (
                            <button key={opt.key} onClick={() => { onChange(opt.key); setOpen(false); }} style={{
                                width: "100%", display: "flex", alignItems: "flex-start", gap: 11,
                                padding: "12px 14px", border: "none", borderTop: `1px solid ${C.border}`,
                                background: isActive ? `${opt.color}0c` : "transparent",
                                cursor: "pointer", textAlign: "left", fontFamily: "inherit",
                                transition: "background .15s",
                            }}
                                onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = `${opt.color}08`; }}
                                onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
                            >
                                <div style={{ width: 32, height: 32, borderRadius: 8, background: `${opt.color}18`, border: `1px solid ${opt.color}28`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 1 }}>
                                    <Activity size={13} color={opt.color} />
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2, flexWrap: "wrap" }}>
                                        <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{opt.name}</span>
                                        <span style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${opt.color}18`, color: opt.color }}>{opt.badge}</span>
                                        {isActive && <span style={{ padding: "1px 7px", borderRadius: 10, fontSize: 9, fontWeight: 700, background: `${opt.color}22`, color: opt.color }}>● Active</span>}
                                    </div>
                                    <div style={{ fontSize: 9, fontFamily: "monospace", color: C.sub, marginBottom: 3, wordBreak: "break-all" }}>{opt.short}</div>
                                    <div style={{ fontSize: 11, color: C.sub, lineHeight: 1.45 }}>{opt.desc}</div>
                                </div>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

// ─── PROFILE MODAL ────────────────────────────────────────────────────────────
// Animation performance note:
// We use `will-change: transform` and `transform: translate3d(...)` to force
// GPU compositing, which eliminates the lag on first open.
const ProfileModal = ({ profile, onSave, onClose }) => {
    const [form, setForm] = useState({ ...profile });
    return (
        <>
            {/* Backdrop — GPU-accelerated opacity fade */}
            <div onClick={onClose} style={{
                position: "fixed", inset: 0,
                background: "rgba(0,0,0,.6)",
                backdropFilter: "blur(6px)",
                WebkitBackdropFilter: "blur(6px)",
                zIndex: 400,
                animation: "fadeIn .2s ease",
                willChange: "opacity",
            }} />

            {/* Modal panel — translate3d triggers GPU layer */}
            <div style={{
                position: "fixed",
                top: "50%", left: "50%",
                transform: "translate3d(-50%, -50%, 0)",
                width: "min(400px, 92vw)",
                background: C.drawerBg, borderRadius: 20, padding: 26,
                zIndex: 410, border: `1px solid ${C.border}`,
                boxShadow: "0 24px 64px rgba(0,0,0,.5)",
                animation: "modalIn .22s cubic-bezier(.34,1.2,.64,1)",
                willChange: "transform, opacity",
            }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
                        <div style={{ width: 42, height: 42, borderRadius: 11, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 800, color: "#fff" }}>
                            {(form.name || "U").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                            <div style={{ fontSize: 15, fontWeight: 700, color: C.text }}>Edit Profile</div>
                            <div style={{ fontSize: 11, color: C.sub }}>Update your details</div>
                        </div>
                    </div>
                    <button onClick={onClose} style={{ width: 28, height: 28, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub }}>
                        <X size={13} />
                    </button>
                </div>

                {/* Fields */}
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {[
                        { label: "Full Name", key: "name", placeholder: "Your name" },
                        { label: "Email", key: "email", placeholder: "you@email.com" },
                        { label: "Role", key: "role", placeholder: "e.g. Recruiter, HR Manager" },
                        { label: "Organisation", key: "org", placeholder: "Company / College" },
                    ].map(({ label, key, placeholder }) => (
                        <div key={key}>
                            <label style={{ fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em", display: "block", marginBottom: 5 }}>{label}</label>
                            <input value={form[key] || ""} onChange={e => setForm(p => ({ ...p, [key]: e.target.value }))} placeholder={placeholder}
                                style={{ width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 9, color: C.text, fontSize: 13, padding: "9px 12px", outline: "none", fontFamily: "inherit", boxSizing: "border-box" }} />
                        </div>
                    ))}
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: 9, marginTop: 20 }}>
                    <button onClick={onClose} style={{ flex: 1, padding: "10px", borderRadius: 10, border: `1px solid ${C.border}`, background: C.inputBg, color: C.sub, cursor: "pointer", fontSize: 13, fontWeight: 600, fontFamily: "inherit" }}>Cancel</button>
                    <button onClick={() => { onSave(form); onClose(); }} style={{ flex: 2, padding: "10px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", gap: 7, fontFamily: "inherit" }}>
                        <Save size={13} /> Save Changes
                    </button>
                </div>
            </div>
        </>
    );
};

// ─── CANDIDATE DRAWER ─────────────────────────────────────────────────────────
const Drawer = ({ candidate: c, onClose, isMobile }) => {
    if (!c) return null;
    const matched = c.matched_skills || [];
    // Use server-computed missing_skills (required skills not found)
    // Fall back to filtering c.skills only if missing_skills not present
    // missing_skills from server = required skills the candidate lacks.
    // Use it always when present (even if empty — means all required skills matched).
    const missing = Array.isArray(c.missing_skills)
        ? c.missing_skills
        : (c.skills || []).filter(s => !matched.includes(s));
    const skScore = Math.round((c.skillScore || 0) * 100);
    const seScore = Math.round((c.semanticScore || 0) * 100);
    const matchedSet = new Set(matched.map(x => x.toLowerCase()));
    const radarSkills = [...new Set([...matched, ...missing])].slice(0, 6);
    const radarData = radarSkills.map(s => ({
        s,
        v: matchedSet.has(s.toLowerCase())
            ? Math.min(100, Math.round(skScore * 0.6 + seScore * 0.4))
            : Math.min(35, Math.round(seScore * 0.35)),
    }));

    return (
        <>
            <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.6)", backdropFilter: "blur(4px)", zIndex: 200 }} />
            <div style={{
                position: "fixed", right: 0, top: 0, bottom: 0,
                width: isMobile ? "100vw" : 420,
                background: C.drawerBg, borderLeft: `1px solid ${C.border}`,
                zIndex: 210, overflowY: "auto", padding: 22,
                animation: "slideInRight .28s cubic-bezier(.4,0,.2,1)",
                display: "flex", flexDirection: "column", gap: 16,
            }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 36, height: 36, borderRadius: 9, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, color: "#fff", flexShrink: 0 }}>
                            {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                        </div>
                        <div>
                            <div style={{ fontSize: 16, fontWeight: 700, color: C.text }}>{c.name}</div>
                            <div style={{ fontSize: 11, color: C.sub }}>{c.filename}</div>
                        </div>
                    </div>
                    <button onClick={onClose} style={{ width: 28, height: 28, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub }}>
                        <X size={13} />
                    </button>
                </div>

                {/* Scores */}
                <div style={card()}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14 }}>Score Breakdown</div>
                    <div style={{ display: "flex", justifyContent: "space-around" }}>
                        {[
                            { label: "Skill", value: Math.round((c.skillScore || 0) * 100), color: C.blue },
                            { label: "Semantic", value: Math.round((c.semanticScore || 0) * 100), color: C.teal },
                            { label: "Final", value: Math.round((c.finalScore || 0) * 100), color: C.amber },
                        ].map(({ label, value, color }) => (
                            <div key={label} style={{ textAlign: "center" }}>
                                <CircularRing value={value} color={color} />
                                <div style={{ fontSize: 10, color: C.sub, marginTop: 7 }}>{label}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Info rows */}
                <div style={card({ padding: "4px 16px" })}>
                    {[
                        { icon: Mail, label: "Email", value: c.email || "Not Found", href: c.email ? `mailto:${c.email}` : null },
                        { icon: Clock, label: "Experience", value: !c.experience ? "Fresher" : `${c.experience} yrs`, href: null },
                        { icon: Phone, label: "Phone", value: c.phone || c.phone_number || c.contact || c.mobile || c.contact_number || "Not found", href: (c.phone || c.phone_number || c.contact || c.mobile || c.contact_number) ? `tel:${(c.phone || c.phone_number || c.contact || c.mobile || c.contact_number || "").replace(/\s/g, "")}` : null },
                    ].map(({ icon: Icon, label, value, href }) => (
                        <div key={label} style={{ display: "flex", gap: 10, padding: "9px 0", borderBottom: `1px solid ${C.border}`, alignItems: "flex-start" }}>
                            <Icon size={13} color={C.sub} style={{ marginTop: 2, flexShrink: 0 }} />
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: 10, color: C.sub }}>{label}</div>
                                {href
                                    ? <a href={href} style={{ fontSize: 13, color: C.blue, marginTop: 1, display: "block", textDecoration: "none", fontWeight: 500 }}
                                        onMouseEnter={e => e.currentTarget.style.textDecoration = "underline"}
                                        onMouseLeave={e => e.currentTarget.style.textDecoration = "none"}>{value}</a>
                                    : <div style={{ fontSize: 13, color: C.cardText, marginTop: 1 }}>{value}</div>
                                }
                            </div>
                        </div>
                    ))}
                    <div style={{ padding: "9px 0" }}>
                        <div style={{ fontSize: 10, color: C.sub, marginBottom: 5 }}>Status</div>
                        <Badge status={c.eligible ? "Shortlisted" : "Rejected"} />
                        {!c.eligible && c.rejection_reason && <div style={{ fontSize: 11, color: "#ef4444", marginTop: 5 }}>{c.rejection_reason}</div>}
                    </div>
                </div>

                {/* Radar chart — analytics-style circular grid with hover tooltip */}
                {radarData.length >= 3 && (
                    <div style={card({ padding: 16 })}>
                        <div style={{ fontSize: 11, fontWeight: 800, color: C.text, marginBottom: 4 }}>Skill Radar</div>
                        <div style={{ fontSize: 10, color: C.muted, marginBottom: 10 }}>Hover each point to see match score</div>
                        <ResponsiveContainer width="100%" height={240}>
                            <RadarChart data={radarData.map(d => ({ skill: d.s, score: d.v, fullMark: 100 }))} outerRadius="68%" margin={{ top:12, right:22, bottom:12, left:22 }}>
                                <PolarGrid gridType="circle" stroke={C.border} />
                                <PolarAngleAxis dataKey="skill" tick={{ fill: C.sub, fontSize: 10, fontWeight: 600 }} />
                                <Tooltip
                                    cursor={false}
                                    contentStyle={{ background: C.drawerBg, border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 12, backdropFilter: "blur(12px)", boxShadow: "0 8px 24px rgba(0,0,0,.4)", color: C.text, padding: "8px 12px" }}
                                    formatter={(value, name) => [`${value}%`, "Match Score"]}
                                    labelStyle={{ color: C.blue, fontWeight: 700, marginBottom: 2 }}
                                />
                                <Radar dataKey="score" stroke={C.blue} strokeWidth={2.5} fill={C.blue} fillOpacity={0.2}
                                    dot={{ fill: C.blue, r: 6, strokeWidth: 2, stroke: C.surface }}
                                    activeDot={{ r: 8, fill: C.amber, stroke: C.text, strokeWidth: 2 }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Skills */}
                <div style={card()}>
                    <div style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 11 }}>Skills</div>
                    {matched.length > 0 && (
                        <div style={{ marginBottom: 10 }}>
                            <div style={{ fontSize: 11, color: C.teal, fontWeight: 600, marginBottom: 6 }}>✓ Matched ({matched.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>{matched.map(s => <SkillChip key={s} label={s} matched />)}</div>
                        </div>
                    )}
                    {missing.length > 0 && (
                        <div>
                            <div style={{ fontSize: 11, color: "#ef4444", fontWeight: 600, marginBottom: 6 }}>✗ Missing ({missing.length})</div>
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>{missing.map(s => <SkillChip key={s} label={s} matched={false} />)}</div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

// ─── UPLOAD VIEW ──────────────────────────────────────────────────────────────
const UploadView = ({ onStartScreening, activeModel, onModelChange, isMobile, backendOnline }) => {
    const [fileItems, setFileItems] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [toast, setToast] = useState(null);
    const [jd, setJd] = useState("");
    const [skills, setSkills] = useState([]);
    const [newSkill, setNewSkill] = useState("");
    const [minExp, setMinExp] = useState(0);
    const [skillWeight, setSkillWeight] = useState(55);
    const fileRef = useRef();
    const jdRef = useRef();

    const MAX_FILES = 30;
    const MAX_SIZE_MB = 10;

    // Pre-fill from saved config.json on mount
    useEffect(() => {
        fetch(`${BASE}/api/config`)
            .then(r => r.json())
            .then(data => {
                if (!data) return;
                if (data.job_description) setJd(data.job_description);
                if (data.required_skills?.length) setSkills(data.required_skills);
                if (data.scoring?.min_experience_years !== undefined)
                    setMinExp(data.scoring.min_experience_years);
                if (data.scoring?.skill_weight !== undefined)
                    setSkillWeight(Math.round(data.scoring.skill_weight * 100));
            })
            .catch(() => { });
    }, []);

    const showToast = (msg, type = "success") => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3000);
    };

    const addFiles = useCallback((incoming) => {
        const arr = Array.from(incoming);
        const valid = arr.filter(f => f.type === "application/pdf");
        const bad = arr.filter(f => f.type !== "application/pdf");
        const tooBig = valid.filter(f => f.size > MAX_SIZE_MB * 1024 * 1024);
        const okSize = valid.filter(f => f.size <= MAX_SIZE_MB * 1024 * 1024);

        if (bad.length) showToast(`${bad.length} non-PDF file(s) skipped`, "error");
        if (tooBig.length) showToast(`${tooBig.length} file(s) exceed ${MAX_SIZE_MB}MB limit`, "error");
        if (!okSize.length) return;

        setFileItems(prev => {
            if (prev.length >= MAX_FILES) { showToast(`Max ${MAX_FILES} resumes per screening`, "error"); return prev; }
            const existing = new Set(prev.map(f => f.name));
            const fresh = okSize.filter(f => !existing.has(f.name)).map(f => ({
                id: `${f.name}-${Date.now()}`,
                name: f.name,
                raw: f,
                size: f.size > 1048576 ? (f.size / 1048576).toFixed(1) + " MB" : (f.size / 1024).toFixed(0) + " KB",
            }));
            if (fresh.length < okSize.length) showToast(`${okSize.length - fresh.length} duplicate(s) skipped`, "error");
            if (fresh.length) {
                showToast(`${fresh.length} PDF(s) added`, "success");
                // Auto-scroll to the JD section after a short delay
                setTimeout(() => jdRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 300);
            }
            return [...prev, ...fresh];
        });
    }, []);

    const onDrop = useCallback(e => {
        e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files);
    }, [addFiles]);

    const canStart = fileItems.length > 0 && jd.trim().length > 10 && backendOnline !== false;
    const activeM = MODELS[activeModel] || MODELS.mpnet;

    // Add skills — supports comma-separated and Enter
    const addSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        setSkills(prev => { const ex = new Set(prev); return [...prev, ...parts.filter(s => !ex.has(s))]; });
        setNewSkill("");
    };
    const handleSkillKeyDown = (e) => { if (e.key === "Enter") { e.preventDefault(); addSkill(); } };

    const handleStart = () => {
        if (!canStart) return;
        onStartScreening({ jd, skills, minExp, skillWeight, fileCount: fileItems.length, rawFiles: fileItems.map(f => f.raw), model: activeModel });
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <div>
                <h2 style={{ fontSize: 21, fontWeight: 800, color: C.text, margin: 0 }}>Upload & Configure</h2>
                <p style={{ color: C.sub, marginTop: 5, fontSize: 13 }}>Add PDF resumes and describe the role. The ML pipeline handles the rest.</p>
            </div>

            {/* Backend offline banner */}
            {backendOnline === false && (
                <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 16px", borderRadius: 12, background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.25)", fontSize: 13, color: "#ef4444" }}>
                    <AlertCircle size={15} style={{ flexShrink: 0 }} />
                    <span>Flask backend is not running. Start it with <code style={{ fontFamily: "monospace", background: "rgba(239,68,68,.12)", padding: "1px 6px", borderRadius: 4 }}>python app.py</code> before screening.</span>
                </div>
            )}

            {/* Drop zone */}
            <div onClick={() => fileRef.current.click()}
                onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                style={{ border: `2px dashed ${isDragging ? C.blue : C.border}`, borderRadius: 18, padding: "36px 28px", cursor: "pointer", textAlign: "center", background: isDragging ? `${C.blue}08` : C.inputBg, display: "flex", flexDirection: "column", alignItems: "center", gap: 10, transition: "all .2s" }}>
                <div style={{ width: 54, height: 54, borderRadius: 14, background: isDragging ? `${C.blue}20` : C.surface, border: `1px solid ${isDragging ? C.blue : C.border}`, display: "flex", alignItems: "center", justifyContent: "center", transition: "all .2s" }}>
                    <Upload size={22} color={isDragging ? C.blue : C.sub} />
                </div>
                <div>
                    <div style={{ fontSize: 15, fontWeight: 600, color: isDragging ? C.blue : C.cardText }}>
                        {isDragging ? "Drop to upload" : "Drag & drop PDF resumes here"}
                    </div>
                    <div style={{ fontSize: 12, color: C.sub, marginTop: 3 }}>
                        or click to browse · multiple files supported
                    </div>
                </div>
                {/* Limits notice */}
                <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
                    {[`Max ${MAX_FILES} files`, `Max ${MAX_SIZE_MB} MB each`, "PDF format only"].map(t => (
                        <span key={t} style={{ fontSize: 10, color: C.muted, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6, padding: "2px 8px" }}>{t}</span>
                    ))}
                </div>
                <input ref={fileRef} type="file" accept=".pdf" multiple hidden onChange={e => addFiles(e.target.files)} />
            </div>

            {/* Uploaded file list */}
            {fileItems.length > 0 && (
                <div style={card({ padding: 0, overflow: "hidden" })}>
                    <div style={{ padding: "10px 15px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>
                            {fileItems.length} file{fileItems.length !== 1 ? "s" : ""} queued
                        </span>
                        <button onClick={() => setFileItems([])} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: C.sub, fontFamily: "inherit" }}>Clear all</button>
                    </div>
                    {fileItems.map(f => (
                        <div key={f.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 15px", borderBottom: `1px solid ${C.border}22` }}>
                            <div style={{ width: 28, height: 28, borderRadius: 7, background: "rgba(239,68,68,.1)", border: "1px solid rgba(239,68,68,.18)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                <FileText size={12} color="#ef4444" />
                            </div>
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ fontSize: 13, color: C.cardText, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</div>
                                <div style={{ fontSize: 10, color: C.sub }}>{f.size}</div>
                            </div>
                            <CheckCircle size={12} color={C.green} />
                            <button onClick={() => setFileItems(p => p.filter(x => x.id !== f.id))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub }}>
                                <X size={11} />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Job requirements — two columns on desktop, same height */}
            <div ref={jdRef} style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left column: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Briefcase size={12} /> Job Description
                    </div>
                    <label style={{ fontSize: 12, color: C.sub, display: "block", marginBottom: 6 }}>Describe the Role</label>
                    <textarea value={jd} onChange={e => { setJd(e.target.value); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe the role — what the candidate will be doing and what you expect from them."
                        style={{ flex: 1, width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 10, color: C.cardText, fontSize: 13, padding: "10px 12px", resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6, boxSizing: "border-box", overflow: "hidden", minHeight: 160 }} />
                </div>

                {/* Right column: Skills + sliders */}
                <div style={card()}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Target size={12} /> Skills & Weights
                    </div>

                    {/* Required Skills header + Clear All */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 12, color: C.sub }}>Required Skills</label>
                        {skills.length > 0 && (
                            <button onClick={() => setSkills([])} style={{ fontSize: 11, color: "#ef4444", background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.2)", borderRadius: 6, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>Clear All</button>
                        )}
                    </div>
                    {skills.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                            {skills.map(s => (
                                <div key={s} style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 20, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, fontSize: 12, fontWeight: 500 }}>
                                    {s}
                                    <button onClick={() => setSkills(p => p.filter(x => x !== s))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub, padding: 0, display: "flex" }}><X size={9} /></button>
                                </div>
                            ))}
                        </div>
                    )}
                    <div style={{ display: "flex", gap: 7, marginBottom: 18 }}>
                        <input value={newSkill} onChange={e => setNewSkill(e.target.value)}
                            onKeyDown={handleSkillKeyDown}
                            placeholder="python, machine learning, sql  — or Enter"
                            style={{ flex: 1, background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 8, color: C.cardText, fontSize: 13, padding: "8px 11px", outline: "none", fontFamily: "inherit" }} />
                        <button onClick={() => addSkill()} style={{ padding: "8px 13px", borderRadius: 8, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, cursor: "pointer", fontFamily: "inherit", fontWeight: 700 }}>
                            <Plus size={13} />
                        </button>
                    </div>

                    {/* Sliders */}
                    {[
                        { label: "Min Experience", value: minExp, set: setMinExp, min: 0, max: 10, step: .5, color: C.amber, fmt: v => v === 0 ? "Freshers — 0 yrs" : `${v}+ yrs` },
                        { label: "Skill Weight", value: skillWeight, set: setSkillWeight, min: 0, max: 100, step: 5, color: C.blue, fmt: v => `${v}% skill / ${100 - v}% semantic` },
                    ].map(({ label, value, set, min, max, step, color, fmt }) => (
                        <div key={label} style={{ marginBottom: 14 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
                                <span style={{ fontSize: 12, color, fontWeight: 700 }}>{fmt(value)}</span>
                            </div>
                            <input type="range" min={min} max={max} step={step} value={value}
                                onChange={e => set(parseFloat(e.target.value))}
                                style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
                        </div>
                    ))}
                </div>
            </div>

            {/*
              ── MODEL INFO PANEL + SWITCHER ───────────────────────────────────
            */}
            <div style={{
                borderRadius: 16, overflow: "hidden",
                border: `1px solid ${activeM.color}30`,
                background: `${activeM.color}06`,
            }}>
                {/* Model info header */}
                <div style={{ padding: "16px 20px", display: "flex", alignItems: "flex-start", gap: 14, borderBottom: `1px solid ${activeM.color}18` }}>
                    {/* Left: icon — vertically centred */}
                    <div style={{ width: 42, height: 42, borderRadius: 11, background: `${activeM.color}18`, border: `1px solid ${activeM.color}30`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, alignSelf: "center" }}>
                        <Activity size={18} color={activeM.color} />
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3, flexWrap: "wrap" }}>
                            <span style={{ fontSize: 14, fontWeight: 800, color: C.text }}>{activeM.name}</span>
                            <span style={{ padding: "2px 8px", borderRadius: 10, fontSize: 10, fontWeight: 700, background: `${activeM.color}18`, color: activeM.color, border: `1px solid ${activeM.color}30` }}>{activeM.badge}</span>
                        </div>
                        <div style={{ fontSize: 11, fontFamily: "monospace", color: C.sub, marginBottom: 5, wordBreak: "break-all" }}>{activeM.short}</div>
                        <div style={{ fontSize: 13, color: C.cardText, lineHeight: 1.55 }}>{activeM.desc}</div>
                        <div style={{ fontSize: 11, color: C.sub, marginTop: 5, lineHeight: 1.5 }}>{activeM.detail}</div>
                    </div>
                </div>

                {/* Inline model switcher — centered */}
                <div style={{ padding: "10px 20px", borderBottom: `1px solid ${activeM.color}14`, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 10, color: C.sub, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em", flexShrink: 0 }}>Switch Model:</span>
                    {Object.values(MODELS).map(m => (
                        <button key={m.key} onClick={() => onModelChange(m.key)}
                            style={{
                                padding: "4px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", fontFamily: "inherit", transition: "all .15s",
                                background: activeModel === m.key ? `${m.color}22` : C.inputBg,
                                color: activeModel === m.key ? m.color : C.sub,
                                border: `1px solid ${activeModel === m.key ? `${m.color}50` : C.border}`,
                            }}>
                            {m.name}
                        </button>
                    ))}
                </div>

                {/* Run button or warning */}
                {canStart ? (
                    <button onClick={handleStart} style={{
                        width: "100%", padding: "16px 24px",
                        border: "none",
                        background: `linear-gradient(135deg,${activeM.color},${activeM.color}99)`,
                        color: "#fff", cursor: "pointer", fontSize: 14, fontWeight: 800,
                        display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                        fontFamily: "inherit", position: "relative", overflow: "hidden",
                        boxShadow: `0 4px 20px ${activeM.color}55`,
                        transition: "box-shadow .2s, transform .1s",
                    }}
                        onMouseEnter={e => { e.currentTarget.style.boxShadow = `0 6px 28px ${activeM.color}88`; e.currentTarget.style.transform = "translateY(-1px)"; }}
                        onMouseLeave={e => { e.currentTarget.style.boxShadow = `0 4px 20px ${activeM.color}55`; e.currentTarget.style.transform = "none"; }}
                        onMouseDown={e => e.currentTarget.style.transform = "scale(.98)"}
                        onMouseUp={e => e.currentTarget.style.transform = "translateY(-1px)"}
                    >
                        <style>{`@keyframes runPulse{0%,100%{opacity:1}50%{opacity:.6}} @keyframes runShimmer{from{transform:translateX(-100%)}to{transform:translateX(200%)}}`}</style>
                        {/* shimmer */}
                        <span style={{ position: "absolute", inset: 0, background: "linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent)", animation: "runShimmer 2.2s ease infinite", pointerEvents: "none" }} />
                        <Zap size={16} style={{ animation: "runPulse 1.6s ease infinite" }} />
                        Run Screening with {activeM.name}
                        <ArrowRight size={15} />
                    </button>
                ) : (
                    <div style={{ padding: "13px 20px", display: "flex", alignItems: "center", justifyContent: "center", gap: 9, fontSize: 12, color: C.sub }}>
                        <Info size={13} color={C.sub} style={{ flexShrink: 0 }} />
                        {backendOnline === false
                            ? "Start the Flask backend (python app.py) before screening."
                            : fileItems.length === 0
                            ? "Upload at least one resume PDF to continue."
                            : "Add a job description (min 10 characters) to continue."}
                    </div>
                )}
            </div>

            {toast && <Toast msg={toast.msg} type={toast.type} onClose={() => setToast(null)} />}
        </div>
    );
};

// ─── PROCESSING VIEW ──────────────────────────────────────────────────────────
const ProcessingView = ({ config, onDone }) => {
    const [progress, setProgress] = useState(0);
    const [step, setStep] = useState(0);
    const [status, setStatus] = useState("Connecting to backend…");
    const [error, setError] = useState(null);
    const [warnings, setWarnings] = useState([]); // parse failures / merge info
    const doneRef = useRef(false);
    const ivRef = useRef(null);

    useEffect(() => {
        if (doneRef.current) return;

        // Animated progress bar — slows near 88% while waiting for the API response
        let current = 0;
        ivRef.current = setInterval(() => {
            current += current < 85 ? 2.0 : 0.1;
            const clamped = Math.min(current, 99);
            setProgress(clamped);
            setStep(Math.min(Math.floor(clamped / 17), PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").length - 1));
        }, 65);

        const run = async () => {
            try {
                // 0. Clear previous results and uploads so old cache never bleeds in
                setStatus("Clearing previous session…");
                await fetch(`${BASE}/api/clear`, { method: "POST" }).catch(() => {});

                // 1. Upload PDFs
                setStatus("Uploading PDFs to backend…");
                const fd = new FormData();
                (config.rawFiles || []).forEach(f => fd.append("files", f));
                const upRes = await fetch(`${BASE}/api/upload-resumes`, { method: "POST", body: fd });
                if (!upRes.ok) {
                    const e = await upRes.json();
                    throw new Error(e.error || "Upload failed");
                }

                // 2. Run ML pipeline
                setStatus("Running ML pipeline — this may take a moment…");
                const screenRes = await fetch(`${BASE}/api/screen`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        job_description: config.jd,
                        required_skills: config.skills || [],
                        model: config.model || "mpnet",
                        config: {
                            min_experience_years: config.minExp || 0,
                            skill_weight: (config.skillWeight || 55) / 100,
                            semantic_weight: 1 - (config.skillWeight || 55) / 100,
                        },
                    }),
                });
                const data = await screenRes.json();
                if (!screenRes.ok) throw new Error(data.error || "Screening failed");

                // Surface parse warnings before handing off results
                const warns = [];
                if (data.parse_failures?.length) {
                    warns.push(`${data.parse_failures.length} PDF(s) could not be parsed and were skipped: ${data.parse_failures.join(", ")}`);
                }
                if (data.merged_count > 0) {
                    warns.push(`${data.merged_count} duplicate resume(s) were merged (same email/phone).`);
                }
                if (warns.length) setWarnings(warns);

                // Done
                clearInterval(ivRef.current);
                doneRef.current = true;
                setProgress(100);
                setStep(PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").length - 1);
                setStatus("Complete!");
                setTimeout(() => onDone(data.results || []), warns.length ? 3000 : 700);

            } catch (err) {
                clearInterval(ivRef.current);
                setError(err.message);
            }
        };

        run();
        return () => clearInterval(ivRef.current);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const pct = Math.round(progress);
    const modelInfo = MODELS[config?.model || "mpnet"];

    // Error screen
    if (error) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh", gap: 14, textAlign: "center", padding: "24px 16px" }}>
                <div style={{ width: 54, height: 54, borderRadius: 15, background: "rgba(239,68,68,.12)", border: "1px solid rgba(239,68,68,.25)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <AlertCircle size={22} color="#ef4444" />
                </div>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.text }}>Pipeline Error</div>
                <div style={{ fontSize: 13, color: "#ef4444", maxWidth: 380, lineHeight: 1.6, background: "rgba(239,68,68,.08)", padding: "10px 16px", borderRadius: 10, border: "1px solid rgba(239,68,68,.2)", wordBreak: "break-word" }}>{error}</div>
                <div style={{ fontSize: 12, color: C.sub }}>Ensure Flask is running: <code style={{ color: C.blue }}>python app.py</code></div>
                <button onClick={() => onDone([])} style={{ padding: "9px 20px", borderRadius: 9, border: `1px solid ${C.border}`, background: C.inputBg, color: C.sub, cursor: "pointer", fontSize: 13, fontFamily: "inherit", fontWeight: 600 }}>
                    ← Back
                </button>
            </div>
        );
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 130px)", padding: "24px 14px" }}>
            <div style={{ width: "100%", maxWidth: 520 }}>
                <div style={{ textAlign: "center", marginBottom: 28 }}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: C.text, marginBottom: 5 }}>Running ML Pipeline</div>
                    <div style={{ fontSize: 13, color: C.sub }}>
                        Analysing <strong style={{ color: C.blue }}>{config?.fileCount || 0}</strong> resume(s) · {status}
                    </div>
                </div>

                {/* Ring + bar */}
                <div style={card({ display: "flex", flexDirection: "column", alignItems: "center", gap: 18, marginBottom: 12 })}>
                    <div style={{ position: "relative", width: 112, height: 112 }}>
                        <svg width={112} height={112} style={{ transform: "rotate(-90deg)" }}>
                            <circle cx={56} cy={56} r={48} fill="none" stroke={`${modelInfo.color}18`} strokeWidth={8} />
                            <circle cx={56} cy={56} r={48} fill="none" stroke={modelInfo.color} strokeWidth={8} strokeLinecap="round"
                                strokeDasharray={2 * Math.PI * 48}
                                strokeDashoffset={2 * Math.PI * 48 * (1 - progress / 100)}
                                style={{ transition: "stroke-dashoffset .12s linear", filter: `drop-shadow(0 0 5px ${modelInfo.color}99)` }}
                            />
                        </svg>
                        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 1 }}>
                            <span style={{ fontSize: 26, fontWeight: 900, color: C.text, lineHeight: 1 }}>{pct}%</span>
                            <span style={{ fontSize: 9, color: C.sub, textTransform: "uppercase", letterSpacing: ".06em" }}>complete</span>
                        </div>
                    </div>
                    <div style={{ width: "100%" }}>
                        <div style={{ height: 5, background: `${modelInfo.color}15`, borderRadius: 3, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${progress}%`, background: `linear-gradient(90deg,${modelInfo.color},${C.teal})`, transition: "width .12s linear", borderRadius: 3 }} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 10, color: C.sub }}>
                            <span>0%</span>
                            <span style={{ color: modelInfo.color, fontWeight: 700, fontFamily: "monospace", fontSize: 9 }}>{modelInfo.short}</span>
                            <span>100%</span>
                        </div>
                    </div>
                </div>

                {/* Parse warnings — shown when some PDFs were skipped or merged */}
                {warnings.length > 0 && (
                    <div style={{ marginBottom: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                        {warnings.map((w, i) => (
                            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 9, padding: "10px 14px", borderRadius: 10, background: "rgba(245,158,11,.08)", border: "1px solid rgba(245,158,11,.25)", fontSize: 12, color: "#f59e0b", lineHeight: 1.5 }}>
                                <AlertCircle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                                {w}
                            </div>
                        ))}
                    </div>
                )}

                {/* Pipeline steps */}
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                    {PIPELINE_STEPS(config?.skillWeight || 55, MODELS[config?.model || "mpnet"]?.name || "MPNet").map((s, i) => {
                        const done = i < step;
                        const active = i === step;
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 13px", borderRadius: 10, background: done ? `${C.green}08` : active ? `${modelInfo.color}0f` : C.inputBg, border: `1px solid ${done ? `${C.green}20` : active ? `${modelInfo.color}26` : C.border}`, transition: "all .3s" }}>
                                <div style={{ width: 19, height: 19, borderRadius: "50%", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", background: done ? `${C.green}18` : active ? `${modelInfo.color}18` : "transparent", border: `1.5px solid ${done ? C.green : active ? modelInfo.color : C.muted}` }}>
                                    {done ? <Check size={9} color={C.green} />
                                        : active ? <div style={{ width: 7, height: 7, borderRadius: "50%", border: `2px solid ${modelInfo.color}`, borderTopColor: "transparent", animation: "spin .7s linear infinite" }} />
                                            : <div style={{ width: 4, height: 4, borderRadius: "50%", background: C.muted }} />}
                                </div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 10, fontFamily: "monospace", color: i <= step ? modelInfo.color : C.muted, fontWeight: 600, marginBottom: 1 }}>{s.label}</div>
                                    <div style={{ fontSize: 11, color: active ? C.cardText : C.muted }}>{s.desc}</div>
                                </div>
                                {done && <CheckCircle size={11} color={C.green} />}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

// ─── DASHBOARD VIEW ───────────────────────────────────────────────────────────
const DashboardView = ({ results, onNav, isMobile, activeModel, onModelChange }) => {
    if (!results || results.length === 0) {
        return (
            <EmptyState icon={LayoutDashboard} title="No results yet"
                sub="Upload resumes and run the ML screening pipeline to see ranked candidates here."
                action={
                    <button onClick={() => onNav("upload")} style={{ padding:"9px 20px", borderRadius:10, border:"none", background:`linear-gradient(135deg,${C.blue},#6366f1)`, color:"#fff", cursor:"pointer", fontSize:13, fontWeight:700, fontFamily:"inherit", display:"flex", alignItems:"center", gap:7 }}>
                        <Upload size={13} /> Upload Resumes
                    </button>
                }
            />
        );
    }

    const eligible  = results.filter(c => c.eligible);
    const rejected  = results.filter(c => !c.eligible);
    const top       = eligible[0] || results[0];
    const avgScore  = eligible.length ? Math.round(eligible.reduce((a,c) => a+(c.finalScore||0),0)/eligible.length*100) : 0;
    const passRate  = results.length ? Math.round(eligible.length/results.length*100) : 0;
    const topScore  = Math.round((top.finalScore||0)*100);
    const TT = { background:"rgba(9,9,11,.96)", border:`1px solid ${C.border}`, borderRadius:10, fontSize:11, color:C.text, backdropFilter:"blur(12px)", boxShadow:"0 8px 24px rgba(0,0,0,.4)" };

    return (
        <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

            {/* KPI strip — liquid glass StatPill */}
            <div style={{ display:"grid", gridTemplateColumns: isMobile?"1fr 1fr":"repeat(4,1fr)", gap:10 }}>
                {[
                    { icon:FileText,   label:"Screened",    value:results.length,  color:C.blue,  sub:"resumes processed"  },
                    { icon:Award,      label:"Shortlisted", value:eligible.length, color:C.green, sub:"eligible candidates" },
                    { icon:TrendingUp, label:"Avg Score",   value:`${avgScore}%`,  color:C.amber, sub:"across shortlisted"  },
                    { icon:Star,       label:"Pass Rate",   value:`${passRate}%`,  color:C.teal,  sub:"of total pool"       },
                ].map(({ icon:Icon, label, value, sub, color }) => (
                    <div key={label} style={{ padding:"14px 16px", borderRadius:14, background:C.surface, border:`1px solid ${C.border}`, position:"relative", overflow:"hidden",
                        transition:"background .15s, border-color .15s" }}
                        onMouseEnter={e=>{ e.currentTarget.style.background=`${color}08`; e.currentTarget.style.borderColor=`${color}28`; }}
                        onMouseLeave={e=>{ e.currentTarget.style.background=C.surface; e.currentTarget.style.borderColor=C.border; }}>
                        <div style={{ position:"absolute", top:0, left:"20%", right:"20%", height:1, background:`linear-gradient(90deg,transparent,${color}44,transparent)` }} />
                        <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
                            <div style={{ width:30, height:30, borderRadius:8, background:`${color}12`, border:`1px solid ${color}20`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                                <Icon size={13} color={color} />
                            </div>
                            <span style={{ fontSize:10, color:C.sub, fontWeight:600 }}>{label}</span>
                        </div>
                        <div style={{ fontSize:22, fontWeight:900, color, lineHeight:1, fontVariantNumeric:"tabular-nums" }}>{value}</div>
                        <div style={{ fontSize:9, color:C.muted, marginTop:3 }}>{sub}</div>
                    </div>
                ))}
            </div>

            {/* Top candidate + score distribution */}
            {eligible.length > 0 && (
                <div style={{ display:"grid", gridTemplateColumns: isMobile?"1fr":"360px 1fr", gap:14, alignItems:"stretch" }}>

                    {/* Top candidate card */}
                    <FieldCard label="Top Candidate" dot={C.amber}>
                        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", textAlign:"center", paddingTop:4 }}>
                            <div style={{ position:"relative", marginBottom:14 }}>
                                <GaugeArc value={topScore} />
                                <div style={{ marginTop:2, fontSize:14, fontWeight:800, color:C.text }}>{top.name}</div>
                                <div style={{ fontSize:10, color:C.muted, marginTop:2 }}>🥇 Rank #1</div>
                            </div>
                            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6, width:"100%", marginBottom:12 }}>
                                {[["Skill",Math.round((top.skillScore||0)*100),C.blue],["Semantic",Math.round((top.semanticScore||0)*100),C.teal]].map(([l,v,col])=>(
                                    <div key={l} style={{ padding:"8px 10px", borderRadius:10, background:`${col}08`, border:`1px solid ${col}16`, textAlign:"center" }}>
                                        <div style={{ fontSize:14, fontWeight:900, color:col }}>{v}%</div>
                                        <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>{l}</div>
                                    </div>
                                ))}
                            </div>
                            {(top.matched_skills||[]).length>0 && (
                                <div style={{ display:"flex", gap:4, overflowX:"auto", scrollbarWidth:"none", width:"100%", justifyContent:"center", flexWrap:"wrap" }}>
                                    {(top.matched_skills||[]).slice(0,5).map(s=><SkillChip key={s} label={s} matched />)}
                                </div>
                            )}
                        </div>
                    </FieldCard>

                    {/* Score distribution — horizontal glass bars */}
                    <FieldCard label="Score Distribution" dot={C.blue}>
                        <div style={{ display:"flex", justifyContent:"flex-end", marginBottom:10 }}>
                            <button onClick={() => onNav("analytics")} style={{ fontSize:10, color:C.blue, background:`${C.blue}10`, border:`1px solid ${C.blue}22`, borderRadius:7, padding:"3px 10px", cursor:"pointer", fontFamily:"inherit", fontWeight:600 }}>
                                Full Analytics →
                            </button>
                        </div>
                        <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                            {[...results].sort((a,b)=>(b.finalScore||0)-(a.finalScore||0)).slice(0,10).map((c,i)=>{
                                const fs = Math.round((c.finalScore||0)*100);
                                const col = fs>=70?C.green:fs>=50?C.blue:C.amber;
                                const medals = ["🥇","🥈","🥉"];
                                return (
                                    <div key={c.id} style={{ display:"flex", alignItems:"center", gap:9 }}>
                                        <div style={{ width:20, textAlign:"center", flexShrink:0, fontSize:i<3?12:9, color:C.muted, lineHeight:1 }}>
                                            {i<3?medals[i]:i+1}
                                        </div>
                                        <div style={{ width:110, fontSize:11, color:C.text, fontWeight:500, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", flexShrink:0 }}>{c.name}</div>
                                        <div style={{ flex:1, height:18, borderRadius:6, overflow:"hidden", background:C.inputBg, border:`1px solid ${C.border}`, backdropFilter:"blur(6px)", position:"relative" }}>
                                            <div style={{ position:"absolute", top:0, left:0, bottom:0, width:`${Math.max(fs,1)}%`, background:`linear-gradient(90deg,${col}77,${col}cc)`, borderRadius:"6px 0 0 6px", transition:"width .8s cubic-bezier(.4,0,.2,1)" }}>
                                                <div style={{ position:"absolute", top:0, left:0, right:0, height:"42%", background:"linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0))", borderRadius:"6px 0 0 0" }} />
                                            </div>
                                            <div style={{ position:"relative", zIndex:1, height:"100%", display:"flex", alignItems:"center", paddingLeft:8 }}>
                                                <span style={{ fontSize:9, fontWeight:700, color:C.text, textShadow:"none" }}>{fs}%</span>
                                            </div>
                                        </div>
                                        <div style={{ width:6, height:6, borderRadius:"50%", flexShrink:0, background:c.eligible?C.green:"#ef4444", opacity:.7 }} />
                                    </div>
                                );
                            })}
                        </div>
                        <div style={{ display:"flex", gap:14, marginTop:12, paddingTop:10, borderTop:`1px solid ${C.border}` }}>
                            {[["● Shortlisted",C.green],["● Rejected","#ef4444"]].map(([l,c])=>(
                                <div key={l} style={{ fontSize:9, color:C.muted }}><span style={{color:c}}>{l.slice(0,1)}</span>{l.slice(1)}</div>
                            ))}
                        </div>
                    </FieldCard>
                </div>
            )}

            {/* Quick actions — glass pill row */}
            <div style={{ borderRadius:16, border:`1px solid ${C.border}`, background:C.surface, backdropFilter:"blur(14px)", padding:"14px 18px" }}>
                <div style={{ fontSize:11, fontWeight:700, color:C.sub, marginBottom:12 }}>Quick Actions</div>
                <div style={{ display:"grid", gridTemplateColumns: isMobile?"1fr 1fr":"repeat(4,1fr)", gap:8 }}>
                    {[
                        { label:"View Rankings", icon:Users,     nav:"candidates", color:C.blue  },
                        { label:"Analytics",     icon:BarChart2, nav:"analytics",  color:C.teal  },
                        { label:"Job Config",    icon:Briefcase, nav:"config",     color:C.amber },
                        { label:"New Screening", icon:RefreshCw, nav:"upload",     color:C.pink  },
                    ].map(({ label, icon:Icon, nav:n, color }) => (
                        <button key={n} onClick={() => onNav(n)} style={{
                            padding:"11px 14px", borderRadius:12, border:`1px solid ${color}22`,
                            background:`${color}08`, color, fontSize:12, fontWeight:600,
                            cursor:"pointer", display:"flex", alignItems:"center", gap:7,
                            fontFamily:"inherit", transition:"all .15s",
                        }}
                            onMouseEnter={e=>{ e.currentTarget.style.background=`${color}14`; e.currentTarget.style.borderColor=`${color}38`; }}
                            onMouseLeave={e=>{ e.currentTarget.style.background=`${color}08`; e.currentTarget.style.borderColor=`${color}22`; }}>
                            <Icon size={13} /> {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* About / ML models panel */}
            <FieldCard label="About This System" dot={C.blue}>
                <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:16 }}>
                    <div style={{ width:38, height:38, borderRadius:10, background:`linear-gradient(135deg,${C.blue},#a78bfa)`, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                        <Sparkles size={16} color="#fff" />
                    </div>
                    <div>
                        <div style={{ fontSize:14, fontWeight:800, color:C.text }}>ML-Based Resume Screening System</div>
                        <div style={{ fontSize:11, color:C.sub }}>By <span style={{color:C.blue,fontWeight:700}}>Madhan Kumar</span> · B.Sc Computer Science Final Year Project</div>
                    </div>
                </div>
                <p style={{ fontSize:12, color:C.cardText, lineHeight:1.7, marginBottom:16 }}>
                    Uses <strong style={{color:C.blue}}>transformer sentence embeddings</strong> to semantically match resumes to job descriptions — beyond keyword matching. <strong style={{color:C.teal}}>Cosine similarity</strong> is computed between resume and JD vectors, combined with skill coverage into a <strong style={{color:C.amber}}>configurable weighted score</strong>.
                </p>
                <div style={{ display:"grid", gridTemplateColumns: isMobile?"1fr":"repeat(3,1fr)", gap:10 }}>
                    {Object.values(MODELS).map(m => (
                        <div key={m.key} style={{ padding:"13px 14px", borderRadius:12, background:`${m.color}07`, border:`1px solid ${m.color}${activeModel===m.key?"40":"18"}`, position:"relative", transition:"border-color .15s" }}
                            onMouseEnter={e=>e.currentTarget.style.borderColor=`${m.color}38`}
                            onMouseLeave={e=>e.currentTarget.style.borderColor=`${m.color}${activeModel===m.key?"40":"18"}`}>
                            {activeModel===m.key && (
                                <div style={{ position:"absolute", top:8, right:8, padding:"1px 7px", borderRadius:10, fontSize:9, fontWeight:700, background:`${m.color}20`, color:m.color }}>● Active</div>
                            )}
                            <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:5 }}>
                                <div style={{ width:7, height:7, borderRadius:"50%", background:m.color, flexShrink:0 }} />
                                <span style={{ fontSize:12, fontWeight:800, color:C.text }}>{m.name}</span>
                                <span style={{ fontSize:8, padding:"1px 6px", borderRadius:10, fontWeight:700, background:`${m.color}16`, color:m.color }}>{m.badge}</span>
                            </div>
                            <div style={{ fontSize:9, fontFamily:"monospace", color:C.muted, marginBottom:5, wordBreak:"break-all" }}>{m.short}</div>
                            <div style={{ fontSize:10, color:C.sub, lineHeight:1.5 }}>{m.desc}</div>
                        </div>
                    ))}
                </div>
            </FieldCard>

        </div>
    );
};

// Rank medal colors
const RANK_COLORS = {
    1: { bg: "rgba(255,215,0,.18)", border: "rgba(255,215,0,.45)", text: "#FFD700" },   // Gold
    2: { bg: "rgba(192,192,192,.18)", border: "rgba(192,192,192,.45)", text: "#C0C0C0" },   // Silver
    3: { bg: "rgba(205,127,50,.18)", border: "rgba(205,127,50,.45)", text: "#CD7F32" },   // Bronze
};

// Export shortlisted candidates — rank, name, email, phone only
function exportToCSV(results) {
    const shortlisted = results.filter(c => c.eligible);
    if (!shortlisted.length) return;

    const headers = ["Rank", "Name", "Email", "Phone"];
    const rows = shortlisted.map(c => [
        c.rank,
        `"${(c.name || "").replace(/"/g, "'")}"`,
        `"${(c.email || "").replace(/"/g, "'")}"`,
        `"${(c.phone || c.phone_number || "").replace(/"/g, "'")}"`,
    ]);

    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "shortlisted_candidates.csv";
    a.click();
    URL.revokeObjectURL(url);
}

const CandidatesView = ({ results, onNav, isMobile }) => {
    const [selected, setSelected] = useState(null);
    const [filter, setFilter] = useState("All");

    if (!results || results.length === 0) {
        return (
            <EmptyState icon={Users} title="No candidates yet"
                sub="Run the screening pipeline first to see ranked candidates here."
                action={
                    <button onClick={() => onNav("upload")} style={{ padding: "9px 20px", borderRadius: 10, border: "none", background: `linear-gradient(135deg,${C.blue},#6366f1)`, color: "#fff", cursor: "pointer", fontSize: 13, fontWeight: 700, fontFamily: "inherit", display: "flex", alignItems: "center", gap: 7 }}>
                        <Zap size={13} /> Start Screening
                    </button>
                }
            />
        );
    }

    const filtered = filter === "All"
        ? results
        : filter === "Shortlisted"
            ? results.filter(c => c.eligible)
            : results.filter(c => !c.eligible);

    return (
        <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: isMobile ? "flex-start" : "center", flexWrap: "wrap", gap: 11, marginBottom: 16 }}>
                <div>
                    <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Candidate Rankings</h2>
                    <p style={{ color: C.sub, marginTop: 3, fontSize: 12 }}>Click any row to view full profile and skill gap analysis.</p>
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {["All", "Shortlisted", "Rejected"].map(f => (
                        <button key={f} onClick={() => setFilter(f)} style={{ padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", background: filter === f ? `${C.blue}20` : C.inputBg, color: filter === f ? C.blue : C.sub, border: `1px solid ${filter === f ? `${C.blue}40` : C.border}`, fontFamily: "inherit" }}>{f}</button>
                    ))}
                    <button onClick={() => exportToCSV(results)} style={{ padding: "5px 12px", borderRadius: 20, fontSize: 11, fontWeight: 600, cursor: "pointer", background: `${C.green}12`, color: C.green, border: `1px solid ${C.green}30`, display: "flex", alignItems: "center", gap: 5, fontFamily: "inherit" }}>
                        <Download size={10} /> Export Shortlisted
                    </button>
                </div>
            </div>

            <div style={card({ padding: 0, overflow: "hidden" })}>
                <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 560 }}>
                        <thead>
                            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                                {["Rank", "Candidate", "Skill", "Semantic", "Final", "Status", ""].map(h => (
                                    <th key={h} style={{ padding: "10px 15px", textAlign: "left", fontSize: 10, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".07em", whiteSpace: "nowrap" }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map((c, i) => (
                                <tr key={c.id} onClick={() => setSelected(c)}
                                    style={{ borderBottom: `1px solid ${C.border}22`, cursor: "pointer", transition: "background .12s" }}
                                    onMouseEnter={e => e.currentTarget.style.background = C.tableFocus}
                                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                                    <td style={{ padding: "12px 15px" }}>
                                        {c.eligible ? (() => {
                                            const medal = RANK_COLORS[c.rank] || { bg: C.inputBg, border: C.border, text: C.sub };
                                            return (
                                                <span style={{ width: 27, height: 27, borderRadius: 7, display: "inline-flex", alignItems: "center", justifyContent: "center", background: medal.bg, border: `1.5px solid ${medal.border}`, fontSize: 11, fontWeight: 800, color: medal.text, boxShadow: c.rank <= 3 ? `0 0 8px ${medal.text}44` : "none" }}>
                                                    {c.rank <= 3 ? ["🥇", "🥈", "🥉"][c.rank - 1] : c.rank}
                                                </span>
                                            );
                                        })() : <span style={{ fontSize: 12, color: C.muted }}>—</span>}
                                    </td>
                                    <td style={{ padding: "12px 15px" }}>
                                        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                                            <div style={{ width: 30, height: 30, borderRadius: 8, background: `linear-gradient(135deg,hsl(${(c.id || 1) * 47},55%,28%),hsl(${(c.id || 1) * 47},55%,38%))`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: `hsl(${(c.id || 1) * 47},80%,80%)`, flexShrink: 0 }}>
                                                {(c.name || "?").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                                            </div>
                                            <div>
                                                <div style={{ fontSize: 13, fontWeight: 600, color: C.text, whiteSpace: "nowrap" }}>{c.name || "Unknown"}</div>
                                                <div style={{ fontSize: 11, color: c.email ? C.sub : C.muted }}>{c.email || "Not Found"}</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.skillScore || 0) * 100)} size={38} sw={3} color={C.blue} /></td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.semanticScore || 0) * 100)} size={38} sw={3} color={C.teal} /></td>
                                    <td style={{ padding: "12px 15px" }}><CircularRing value={Math.round((c.finalScore || 0) * 100)} size={38} sw={3} color={C.amber} /></td>
                                    <td style={{ padding: "12px 15px" }}><Badge status={c.eligible ? "Shortlisted" : "Rejected"} /></td>
                                    <td style={{ padding: "12px 15px" }}><ChevronRight size={13} color={C.muted} /></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <Drawer candidate={selected} onClose={() => setSelected(null)} isMobile={isMobile} />
        </div>
    );
};

// ─── JOB CONFIG VIEW ──────────────────────────────────────────────────────────
const JobConfigView = () => {
    const [saved, setSaved] = useState(false);
    const [cfg, setCfg] = useState({ jd: "", skills: [], minExp: 0, skillW: 55, semanticW: 45 });
    const [newSkill, setNewSkill] = useState("");

    useEffect(() => {
        fetch(`${BASE}/api/config`)
            .then(r => r.json())
            .then(data => {
                if (data) setCfg(prev => ({
                    ...prev,
                    jd: data.job_description || "",
                    skills: data.required_skills || [],
                    skillW: data.scoring?.skill_weight ? Math.round(data.scoring.skill_weight * 100) : 55,
                    semanticW: data.scoring?.semantic_weight ? Math.round(data.scoring.semantic_weight * 100) : 45,
                    minExp: data.scoring?.min_experience_years || 0,
                }));
            })
            .catch(() => { });
    }, []);

    const addCfgSkill = (raw = newSkill) => {
        const parts = raw.split(",").map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
        setCfg(prev => { const ex = new Set(prev.skills); return { ...prev, skills: [...prev.skills, ...parts.filter(s => !ex.has(s))] }; });
        setNewSkill("");
    };

    const save = async () => {
        try {
            await fetch(`${BASE}/api/config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_description: cfg.jd,
                    required_skills: cfg.skills,
                    scoring: {
                        skill_weight: cfg.skillW / 100,
                        semantic_weight: cfg.semanticW / 100,
                        min_experience_years: cfg.minExp,
                        top_n: 20,
                    },
                }),
            });
        } catch (_) { }
        setSaved(true);
        setTimeout(() => setSaved(false), 2500);
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Header + What is this */}
            <div>
                <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Job Configuration</h2>
                <p style={{ color: C.sub, marginTop: 4, fontSize: 12 }}>
                    Saved values are written to <code style={{ color: C.teal, background: `${C.teal}15`, padding: "1px 6px", borderRadius: 4, fontSize: 11 }}>config.json</code> and become the default pre-fill for every new Upload session.
                </p>
            </div>

            {/* What is this used for */}
            <div style={card({ background: `${C.blue}06`, borderColor: `${C.blue}22`, padding: 16 })}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 9 }}>
                    <Info size={13} color={C.blue} />
                    <span style={{ fontSize: 12, fontWeight: 700, color: C.blue }}>What is Job Config for?</span>
                </div>
                <p style={{ fontSize: 12, color: C.cardText, lineHeight: 1.7, margin: 0 }}>
                    Think of this as your <strong style={{ color: C.text }}>saved job profile</strong>. When you screen multiple batches of resumes for the same role over several days, you don't need to retype the job description and skills each time — just save them here once and they auto-fill in the Upload tab. It also lets you standardise the scoring weights (skill vs. semantic) for your organisation.
                </p>
            </div>

            {/* Two-column layout same as Upload */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, alignItems: "stretch" }}>
                {/* Left: JD */}
                <div style={{ ...card(), display: "flex", flexDirection: "column" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Briefcase size={12} /> Job Description
                    </div>
                    <textarea value={cfg.jd} onChange={e => { setCfg(p => ({ ...p, jd: e.target.value })); e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
                        placeholder="Describe the role — what the candidate will be doing and what you expect from them."
                        style={{ flex: 1, width: "100%", background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 10, color: C.cardText, fontSize: 13, padding: "10px 12px", resize: "none", outline: "none", fontFamily: "inherit", lineHeight: 1.6, boxSizing: "border-box", overflow: "hidden", minHeight: 160 }} />
                </div>

                {/* Right: Skills + Weights */}
                <div style={card()}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.sub, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 7 }}>
                        <Target size={12} /> Skills & Weights
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
                        <label style={{ fontSize: 12, color: C.sub }}>Required Skills</label>
                        {cfg.skills.length > 0 && (
                            <button onClick={() => setCfg(p => ({ ...p, skills: [] }))} style={{ fontSize: 11, color: "#ef4444", background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.2)", borderRadius: 6, padding: "2px 9px", cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>Clear All</button>
                        )}
                    </div>
                    {cfg.skills.length > 0 && (
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                            {cfg.skills.map(s => (
                                <div key={s} style={{ display: "flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 20, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, fontSize: 12, fontWeight: 500 }}>
                                    {s}
                                    <button onClick={() => setCfg(p => ({ ...p, skills: p.skills.filter(x => x !== s) }))} style={{ background: "none", border: "none", cursor: "pointer", color: C.sub, padding: 0, display: "flex" }}><X size={9} /></button>
                                </div>
                            ))}
                        </div>
                    )}
                    <div style={{ display: "flex", gap: 7, marginBottom: 18 }}>
                        <input value={newSkill} onChange={e => setNewSkill(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addCfgSkill(); } }}
                            placeholder="python, machine learning, sql  — or Enter"
                            style={{ flex: 1, background: C.inputBg, border: `1px solid ${C.border}`, borderRadius: 8, color: C.cardText, fontSize: 13, padding: "8px 11px", outline: "none", fontFamily: "inherit" }} />
                        <button onClick={() => addCfgSkill()} style={{ padding: "8px 13px", borderRadius: 8, background: `${C.blue}18`, border: `1px solid ${C.blue}30`, color: C.blue, cursor: "pointer", fontFamily: "inherit", fontWeight: 700 }}>
                            <Plus size={13} />
                        </button>
                    </div>

                    {[
                        { label: "Min Experience", key: "minExp", min: 0, max: 10, step: .5, color: C.amber, fmt: v => v === 0 ? "Freshers — 0 yrs" : `${v}+ yrs` },
                        { label: "Skill Weight", key: "skillW", min: 0, max: 100, step: 5, color: C.blue, fmt: v => `${v}% skill / ${100 - v}% semantic` },
                    ].map(({ label, key, min, max, step, color, fmt }) => (
                        <div key={key} style={{ marginBottom: 14 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                                <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
                                <span style={{ fontSize: 12, color, fontWeight: 700 }}>{fmt(cfg[key])}</span>
                            </div>
                            <input type="range" min={min} max={max} step={step} value={cfg[key]}
                                onChange={e => setCfg(p => ({ ...p, [key]: parseFloat(e.target.value), ...(key === "skillW" ? { semanticW: 100 - parseFloat(e.target.value) } : {}) }))}
                                style={{ width: "100%", accentColor: color, cursor: "pointer" }} />
                        </div>
                    ))}
                </div>
            </div>

            {/* Centered save button */}
            <div style={{ display: "flex", justifyContent: "center" }}>
                <button onClick={save} style={{ padding: "12px 36px", borderRadius: 12, border: `1px solid ${saved ? `${C.green}40` : `${C.blue}40`}`, background: saved ? `${C.green}12` : `linear-gradient(135deg,${C.blue}18,#a78bfa18)`, color: saved ? C.green : C.blue, cursor: "pointer", fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 9, fontFamily: "inherit", transition: "all .3s", boxShadow: saved ? `0 0 16px ${C.green}22` : `0 0 16px ${C.blue}18` }}>
                    {saved ? <><CheckCircle size={15} /> Saved to config.json</> : <><Save size={15} /> Save Configuration</>}
                </button>
            </div>
        </div>
    );
};


// ─── ANALYTICS — liquid glass design ────────────────────────────────

// Fieldset card: title cut into border like <fieldset><legend>
const FieldCard = ({ label, dot, children, xtra = {} }) => (
    <div style={{
        position: "relative",
        border: `1px solid ${C.border}`,
        borderRadius: 16,
        padding: "30px 20px 20px",
        background: C.surface,
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        ...xtra,
    }}>
        <div style={{
            position: "absolute", top: -11, left: 16,
            display: "flex", alignItems: "center", gap: 6,
            background: C.bg, padding: "0 8px",
        }}>
            {dot && <div style={{ width: 6, height: 6, borderRadius: "50%", background: dot, flexShrink: 0 }} />}
            <span style={{ fontSize: 12, fontWeight: 800, color: C.text, letterSpacing: ".03em" }}>{label}</span>
        </div>
        {children}
    </div>
);

// Subtle liquid bar — no glow, just a soft fill with top sheen
const SoftBar = ({ pct, color, h = 6 }) => (
    <div style={{ width: "100%", height: h, background: C.inputBg, borderRadius: h, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${Math.max(pct, 1)}%`, background: color, borderRadius: h, opacity: 0.85, transition: "width .8s cubic-bezier(.4,0,.2,1)", position: "relative" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "40%", background: "rgba(255,255,255,0.18)", borderRadius: h }} />
        </div>
    </div>
);

// Compact stat — small version of the KPI card
const StatPill = ({ label, value, color, sub }) => (
    <div style={{ padding: "12px 14px", borderRadius: 12, background: C.surface, border: `1px solid ${C.border}`, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, left: "25%", right: "25%", height: 1, background: `linear-gradient(90deg,transparent,${color}55,transparent)` }} />
        <div style={{ fontSize: 20, fontWeight: 900, color, lineHeight: 1, fontVariantNumeric: "tabular-nums" }}>{value}</div>
        <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginTop: 4 }}>{label}</div>
        {sub && <div style={{ fontSize: 9, color: C.muted, marginTop: 1 }}>{sub}</div>}
    </div>
);

// iOS 26–style liquid glass tab bar
const LiquidTabBar = ({ sections, active, onChange }) => (
    <div style={{
        display: "inline-flex", alignSelf: "flex-start",
        background: C.inputBg,
        backdropFilter: "blur(24px)", WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(255,255,255,0.09)",
        borderRadius: 14, padding: 3, gap: 2,
        boxShadow: "0 2px 12px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.07)",
    }}>
        {sections.map(s => {
            const on = active === s.id;
            return (
                <button key={s.id} onClick={() => onChange(s.id)} style={{
                    padding: "6px 15px", borderRadius: 11, border: "none",
                    background: on ? C.surface : "transparent",
                    color: on ? C.text : C.sub,
                    fontSize: 12, fontWeight: on ? 700 : 500,
                    cursor: "pointer", fontFamily: "inherit",
                    transition: "all .18s cubic-bezier(.4,0,.2,1)",
                    whiteSpace: "nowrap",
                    boxShadow: on ? "0 1px 6px rgba(0,0,0,.15)" : "none",
                }}>{s.label}</button>
            );
        })}
    </div>
);

// ─── ANALYTICS VIEW ───────────────────────────────────────────────────────────
const AnalyticsView = ({ results, isMobile, onNav, activeModel }) => {
    const [tab, setTab] = useState("overview");
    const [evalMetrics, setEvalMetrics] = useState(null);

    useEffect(() => {
        fetch(`${BASE}/metrics/latest`)
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data && !data.error) setEvalMetrics(data); })
            .catch(err => console.error("[AnalyticsView] Failed to load /metrics/latest:", err));
    }, [results]);

    if (!results || results.length === 0)
        return <EmptyState icon={BarChart2} title="No analytics data" sub="Run the screening pipeline first." />;

    const eligible  = results.filter(c => c.eligible);
    const rejected  = results.filter(c => !c.eligible);
    const top3      = eligible.slice(0, 3);
    const borderline = eligible.filter(c => { const s = Math.round((c.finalScore||0)*100); return s >= 50 && s <= 65; });

    const avg = (arr, key) => arr.length ? Math.round(arr.reduce((a,c) => a+(c[key]||0), 0) / arr.length * 100) : 0;
    const avgFinal = avg(eligible, "finalScore");
    const avgSkill = avg(eligible, "skillScore");
    const avgSem   = avg(eligible, "semanticScore");
    const passRate = results.length ? Math.round(eligible.length / results.length * 100) : 0;
    const topScore = eligible.length ? Math.round((eligible[0].finalScore||0)*100) : 0;

    // Skill maps
    const skillCount = {}, missingCount = {};
    results.forEach(c => {
        (c.matched_skills||[]).forEach(s => { skillCount[s] = (skillCount[s]||0)+1; });
        (c.missing_skills||[]).forEach(s => { missingCount[s] = (missingCount[s]||0)+1; });
    });
    const coverageData = Object.keys({ ...skillCount, ...missingCount }).map(sk => ({
        skill: sk,
        matched: skillCount[sk]||0,
        pct: Math.round(((skillCount[sk]||0)/results.length)*100),
    })).sort((a,b) => b.pct - a.pct);

    const gapData = Object.entries(missingCount).map(([skill,count]) => ({
        skill, count, pct: Math.round(count/results.length*100)
    })).sort((a,b) => b.count - a.count);

    // Buckets
    const buckets = {"0–20":0,"21–40":0,"41–60":0,"61–80":0,"81–100":0};
    results.forEach(c => {
        const s = Math.round((c.finalScore||0)*100);
        if (s<=20) buckets["0–20"]++;
        else if (s<=40) buckets["21–40"]++;
        else if (s<=60) buckets["41–60"]++;
        else if (s<=80) buckets["61–80"]++;
        else buckets["81–100"]++;
    });

    // Quadrants
    const quads = { tr:[], br:[], tl:[], bl:[] };
    results.forEach(c => {
        const sk = c.skillScore||0, se = c.semanticScore||0;
        if (sk>=0.5 && se>=0.45) quads.tr.push(c);
        else if (sk>=0.5)        quads.br.push(c);
        else if (se>=0.45)       quads.tl.push(c);
        else                     quads.bl.push(c);
    });

    const TABS = [
        { id:"overview",  label:"Overview"  },
        { id:"funnel",    label:"Funnel"    },
        { id:"talent",    label:"Talent"    },
        { id:"skills",    label:"Skills"    },
        { id:"decisions", label:"Decisions" },
    ];

    const TT = { background: "rgba(9,9,11,.96)", border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 11, color: C.text, backdropFilter: "blur(12px)", boxShadow: "0 8px 24px rgba(0,0,0,.4)" };

    return (
        <div style={{ display:"flex", flexDirection:"column", gap: 18 }}>

            {/* Header */}
            <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", flexWrap:"wrap", gap: 12 }}>
                <div>
                    <div style={{ fontSize: 20, fontWeight: 900, color: C.text, letterSpacing:"-.02em" }}>Recruitment Analytics</div>
                    <div style={{ fontSize: 11, color: C.sub, marginTop: 3 }}>
                        {results.length} screened · {eligible.length} shortlisted · {rejected.length} rejected
                    </div>
                </div>
                <LiquidTabBar sections={TABS} active={tab} onChange={setTab} />
            </div>

            {/* ═══ OVERVIEW ═══════════════════════════════════════════════════ */}
            {tab === "overview" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>

                    {/* Compact KPI row */}
                    <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(5,1fr)", gap: 10 }}>
                        {[
                            { label:"Screened",    value: results.length,  color: C.blue,   sub:"total resumes"    },
                            { label:"Shortlisted", value: eligible.length, color: C.green,  sub:"passed filters"   },
                            { label:"Rejected",    value: rejected.length, color:"#ef4444", sub:"did not qualify"  },
                            { label:"Pass Rate",   value:`${passRate}%`,   color: C.teal,   sub:"of total pool"    },
                            { label:"Top Score",   value:`${topScore}%`,   color: C.amber,  sub: eligible[0]?.name?.split(" ")[0] || "—" },
                        ].map(p => <StatPill key={p.label} {...p} />)}
                    </div>

                    {/* Score distribution — glass bars, full names, no # prefix */}
                    <FieldCard label="Score Distribution" dot={C.blue} xtra={{ padding:"30px 14px 14px" }}>
                        <div style={{ display:"flex", flexDirection:"column", gap: 7 }}>
                            {[...results].sort((a,b)=>(b.finalScore||0)-(a.finalScore||0)).map((c, i) => {
                                const fs = Math.round((c.finalScore||0)*100);
                                const sk = Math.round((c.skillScore||0)*100);
                                const se = Math.round((c.semanticScore||0)*100);
                                const col = fs >= 70 ? C.green : fs >= 50 ? C.blue : C.amber;
                                const medals = ["🥇","🥈","🥉"];
                                return (
                                    <div key={c.id} style={{ display:"flex", alignItems:"center", gap: 10 }}>
                                        {/* Rank — medal or number */}
                                        <div style={{ width: 22, textAlign:"center", flexShrink: 0, fontSize: i < 3 ? 13 : 10, fontWeight: 700, color: C.muted, lineHeight: 1 }}>
                                            {i < 3 ? medals[i] : i + 1}
                                        </div>
                                        {/* Full name */}
                                        <div style={{ width: isMobile ? 80 : 130, fontSize: 11, fontWeight: 500, color: C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", flexShrink: 0 }}>
                                            {c.name || "Unknown"}
                                        </div>
                                        {/* Liquid glass bar — frosted track, saturated fill, inner sheen only on fill */}
                                        <div style={{ flex: 1, height: 20, borderRadius: 7, overflow:"hidden", background:C.inputBg, border:`1px solid ${C.border}`, backdropFilter:"blur(6px)", WebkitBackdropFilter:"blur(6px)", position:"relative" }}>
                                            {/* Fill */}
                                            <div style={{ position:"absolute", top:0, left:0, bottom:0, width:`${Math.max(fs,1)}%`, background:`linear-gradient(90deg,${col}88,${col}dd)`, transition:"width .85s cubic-bezier(.4,0,.2,1)", borderRadius: "7px 0 0 7px" }}>
                                                {/* Top sheen — only on fill area */}
                                                <div style={{ position:"absolute", top:0, left:0, right:0, height:"42%", background:"linear-gradient(180deg,rgba(255,255,255,0.18),rgba(255,255,255,0))", borderRadius:"7px 0 0 0" }} />
                                                {/* Bottom inner shadow on fill */}
                                                <div style={{ position:"absolute", bottom:0, left:0, right:0, height:"30%", background:"rgba(0,0,0,0.15)", borderRadius:"0 0 0 7px" }} />
                                            </div>
                                            {/* Text overlay */}
                                            <div style={{ position:"relative", zIndex:1, height:"100%", display:"flex", alignItems:"center", paddingLeft:10, gap:6 }}>
                                                <span style={{ fontSize:10, fontWeight:800, color:C.text, textShadow:"none", fontVariantNumeric:"tabular-nums" }}>{fs}%</span>

                                            </div>
                                        </div>
                                        {/* Status dot */}
                                        <div style={{ width: 6, height: 6, borderRadius:"50%", flexShrink:0, background: c.eligible ? C.green : "#ef4444", opacity:.75 }} />
                                    </div>
                                );
                            })}
                        </div>
                        <div style={{ display:"flex", gap:14, marginTop:10, paddingTop:8, borderTop:`1px solid ${C.border}` }}>
                            {[["● Shortlisted","#22c55e"],["● Rejected","#ef4444"]].map(([l,c])=>(
                                <div key={l} style={{ display:"flex", alignItems:"center", gap:4, fontSize:9, color:C.muted }}>
                                    <span style={{color:c}}>●</span>{l.slice(1)}
                                </div>
                            ))}
                        </div>
                    </FieldCard>

                    {/* Model Evaluation Metrics */}
                    {evalMetrics && (!activeModel || evalMetrics.model === activeModel) && (
                        <FieldCard label="Model Evaluation Metrics" dot={C.teal} xtra={{ padding:"30px 14px 16px" }}>
                            <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                                {/* Model badge row */}
                                <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4 }}>
                                    <span style={{ fontSize:10, fontWeight:700, color:C.muted, textTransform:"uppercase", letterSpacing:".06em" }}>Model</span>
                                    <span style={{ fontSize:11, fontWeight:800, color:C.teal, background:`${C.teal}12`, border:`1px solid ${C.teal}28`, borderRadius:6, padding:"1px 9px" }}>
                                        {evalMetrics.model || "—"}
                                    </span>
                                    <span style={{ fontSize:9, color:C.muted }}>
                                        {evalMetrics.candidates} candidates · threshold {Math.round((evalMetrics.threshold||0.60)*100)}%
                                    </span>
                                </div>
                                {/* Metric chips grid */}
                                <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(4,1fr)", gap:10 }}>
                                    {[
                                        { label:"Accuracy",  value: evalMetrics.accuracy,  color: C.teal  },
                                        { label:"Precision", value: evalMetrics.precision, color: C.blue  },
                                        { label:"Recall",    value: evalMetrics.recall,    color: C.green },
                                        { label:"F1 Score",  value: evalMetrics.f1,        color: C.amber },
                                    ].map(({ label, value, color }) => {
                                        const pct = value != null ? Math.round(value * 100) : null;
                                        return (
                                            <div key={label} style={{ textAlign:"center", padding:"14px 8px", borderRadius:12, background:`${color}08`, border:`1px solid ${color}22`, position:"relative", overflow:"hidden" }}>
                                                <div style={{ position:"absolute", top:0, left:"20%", right:"20%", height:1, background:`linear-gradient(90deg,transparent,${color}55,transparent)` }} />
                                                <div style={{ fontSize:22, fontWeight:900, color, fontVariantNumeric:"tabular-nums", lineHeight:1 }}>
                                                    {pct != null ? `${pct}%` : "—"}
                                                </div>
                                                <div style={{ fontSize:10, fontWeight:600, color:C.text, marginTop:5 }}>{label}</div>
                                                {pct != null && <SoftBar pct={pct} color={color} h={3} />}
                                            </div>
                                        );
                                    })}
                                </div>
                                {/* Similarity stats row */}
                                {(evalMetrics.mean_similarity != null || evalMetrics.std_similarity != null) && (
                                    <div style={{ display:"flex", gap:12, marginTop:2, padding:"8px 10px", borderRadius:9, background:C.inputBg, border:`1px solid ${C.border}` }}>
                                        <div style={{ fontSize:9, color:C.muted, fontWeight:700, textTransform:"uppercase", letterSpacing:".06em", alignSelf:"center" }}>Similarity</div>
                                        {evalMetrics.mean_similarity != null && (
                                            <div style={{ fontSize:11, color:C.cardText }}>
                                                mean <span style={{fontWeight:800,color:C.text}}>{Math.round(evalMetrics.mean_similarity*100)}%</span>
                                            </div>
                                        )}
                                        {evalMetrics.std_similarity != null && (
                                            <div style={{ fontSize:11, color:C.cardText }}>
                                                std <span style={{fontWeight:800,color:C.text}}>{Math.round(evalMetrics.std_similarity*100)}%</span>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </FieldCard>
                    )}

                    {/* Score Bands — full width horizontal strip, matches score distribution width */}
                    <FieldCard label="Score Bands" dot={C.amber} xtra={{ padding:"30px 14px 16px" }}>
                        <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1px 2fr", gap:0, alignItems:"start" }}>
                            {/* Left: formula + avg stats */}
                            <div style={{ paddingRight:16 }}>
                                <div style={{ padding:"9px 12px", borderRadius:9, background:C.inputBg, border:`1px solid ${C.border}`, marginBottom:12 }}>
                                    <div style={{ fontSize:9, color:C.muted, marginBottom:4, fontWeight:700, textTransform:"uppercase", letterSpacing:".06em" }}>Score Formula</div>
                                    <div style={{ fontSize:10, color:C.cardText, fontFamily:"monospace", lineHeight:1.7 }}>
                                        Final = (skill_weight × skill_score) + (sem_weight × semantic_score)
                                    </div>
                                    <div style={{ fontSize:9, color:C.muted, marginTop:3 }}>
                                        Default: <span style={{color:C.blue,fontWeight:700}}>55% skill</span> + <span style={{color:C.teal,fontWeight:700}}>45% semantic</span>
                                    </div>
                                </div>
                                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6 }}>
                                    {[["Skill",avgSkill,C.blue],["Semantic",avgSem,C.teal],["Final",avgFinal,C.amber]].map(([l,v,col])=>(
                                        <div key={l} style={{ textAlign:"center", padding:"8px 4px", borderRadius:9, background:`${col}08`, border:`1px solid ${col}18` }}>
                                            <div style={{ fontSize:15, fontWeight:900, color:col, fontVariantNumeric:"tabular-nums" }}>{v}%</div>
                                            <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>avg {l}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            {/* Divider */}
                            <div style={{ background:C.border, alignSelf:"stretch", margin:"0 4px" }} />
                            {/* Right: 5 bands */}
                            <div style={{ display:"flex", flexDirection:"column", gap:10, paddingLeft:16 }}>
                                {[
                                    { range:"81–100", label:"Excellent", color:"#10b981" },
                                    { range:"61–80",  label:"Good",      color:"#6366f1" },
                                    { range:"41–60",  label:"Average",   color:"#f59e0b" },
                                    { range:"21–40",  label:"Low",       color:"#f97316" },
                                    { range:"0–20",   label:"Very Low",  color:"#ef4444" },
                                ].map(({ range, label, color }) => {
                                    const count = buckets[range]||0;
                                    const pct   = results.length ? Math.round(count/results.length*100) : 0;
                                    return (
                                        <div key={range}>
                                            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                                                <span style={{ fontSize:11, color:count?C.text:C.muted, fontWeight:600 }}>{label} <span style={{color:C.muted,fontWeight:400,fontSize:9}}>{range}</span></span>
                                                <span style={{ fontSize:11, fontWeight:800, color:count?color:C.muted, fontVariantNumeric:"tabular-nums" }}>{count}</span>
                                            </div>
                                            <SoftBar pct={pct} color={color} h={6} />
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </FieldCard>

                    {/* Top Candidates — full width, 3-column card grid */}
                    {top3.length > 0 && (
                        <FieldCard label="Top Candidates" dot={C.amber} xtra={{ padding:"30px 14px 14px" }}>
                            <div style={{ display:"grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3,1fr)", gap:12 }}>
                                {top3.map((c, i) => {
                                    const glow  = [C.amber, "#94a3b8", "#cd7f32"][i];
                                    const medal = ["🥇","🥈","🥉"][i];
                                    const fs    = Math.round((c.finalScore||0)*100);
                                    const sk    = Math.round((c.skillScore||0)*100);
                                    const se    = Math.round((c.semanticScore||0)*100);
                                    return (
                                        <div key={c.id} style={{ borderRadius:14, background:`${glow}06`, border:`1px solid ${glow}20`, overflow:"hidden", position:"relative", display:"flex", flexDirection:"column" }}>
                                            <div style={{ position:"absolute", top:0, left:0, right:0, height:1, background:`linear-gradient(90deg,transparent,${glow}55,transparent)` }} />
                                            {/* Header */}
                                            <div style={{ display:"flex", alignItems:"center", gap:11, padding:"14px 14px 10px" }}>
                                                <div style={{ width:36, height:36, borderRadius:9, background:`linear-gradient(135deg,${glow}33,${glow}18)`, border:`1px solid ${glow}30`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:800, color:glow, flexShrink:0 }}>
                                                    {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                                </div>
                                                <div style={{ flex:1, minWidth:0 }}>
                                                    <div style={{ fontSize:12, fontWeight:800, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.name}</div>
                                                    <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>{medal} Rank #{i+1}</div>
                                                </div>
                                                <div style={{ fontSize:20, fontWeight:900, color:glow, flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{fs}%</div>
                                            </div>
                                            {/* Score bar */}
                                            <div style={{ padding:"0 14px 10px" }}>
                                                <SoftBar pct={fs} color={glow} h={5} />
                                            </div>
                                            {/* Sk / Se chips */}
                                            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6, padding:"0 14px 12px" }}>
                                                {[["Skill",sk,C.blue],["Semantic",se,C.teal]].map(([l,v,col])=>(
                                                    <div key={l} style={{ textAlign:"center", padding:"6px 8px", borderRadius:8, background:`${col}08`, border:`1px solid ${col}16` }}>
                                                        <div style={{ fontSize:13, fontWeight:800, color:col }}>{v}%</div>
                                                        <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>{l}</div>
                                                    </div>
                                                ))}
                                            </div>
                                            {/* Matched skills */}
                                            {(c.matched_skills||[]).length>0 && (
                                                <div style={{ padding:"0 14px 14px", display:"flex", flexWrap:"wrap", gap:4, flex:1 }}>
                                                    {(c.matched_skills||[]).map(s=>(
                                                        <span key={s} style={{ fontSize:9, padding:"2px 8px", borderRadius:20, background:`${glow}10`, color:glow, border:`1px solid ${glow}20`, fontWeight:600 }}>{s}</span>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>
                    )}
                </div>
            )}

            {/* ═══ FUNNEL ══════════════════════════════════════════════════════ */}
            {tab === "funnel" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>
                    <FieldCard label="Recruitment Funnel" dot={C.blue}>
                        <p style={{ fontSize:12, color:C.cardText, marginBottom:18, lineHeight:1.6 }}>
                            Tracks how many candidates progressed through each ML pipeline stage — from raw PDF upload to final shortlist.
                        </p>
                        {(() => {
                            const stages = [
                                { label:"PDFs Uploaded",       value:results.length,  color:C.blue,    desc:"Raw resumes received"                      },
                                { label:"Successfully Parsed", value:results.length,  color:"#818cf8", desc:"Text extracted via pdfplumber / PyMuPDF"    },
                                { label:"Profiles Extracted",  value:results.length,  color:C.teal,    desc:"Name · email · skills · experience via NER" },
                                { label:"Scored & Ranked",     value:results.length,  color:C.amber,   desc:`Avg final: ${avgFinal}% · skill: ${avgSkill}% · semantic: ${avgSem}%` },
                                { label:"Shortlisted",         value:eligible.length, color:C.green,   desc:`Pass rate: ${passRate}% of total pool`       },
                            ];
                            return stages.map((s, i) => {
                                const pct  = Math.round(s.value / (stages[0].value||1) * 100);
                                const drop = i>0 ? stages[i-1].value - s.value : 0;
                                return (
                                    <div key={s.label}>
                                        {drop > 0 && (
                                            <div style={{ display:"flex", alignItems:"center", gap:0, padding:"2px 0 2px 16px" }}>
                                                {/* Vertical connector aligned with the left border accent */}
                                                <div style={{ width:2, height:20, background:`rgba(239,68,68,0.3)`, borderRadius:2, flexShrink:0 }} />
                                                <div style={{ marginLeft:12, display:"flex", alignItems:"center", gap:5 }}>
                                                    <span style={{ fontSize:9, color:"#ef4444", fontWeight:700, letterSpacing:".02em" }}>↓ {drop} candidate{drop>1?"s":""} dropped</span>
                                                </div>
                                            </div>
                                        )}
                                        <div style={{ display:"flex", alignItems:"center", gap:14, padding:"14px 16px", borderRadius:13, background:`${s.color}05`, border:`1px solid ${s.color}16`, marginBottom:2, position:"relative", overflow:"hidden" }}>
                                            <div style={{ position:"absolute", top:0, left:0, bottom:0, width:2, background:s.color, opacity:.55, borderRadius:2 }} />
                                            {/* Step number badge */}
                                            <div style={{ width:26, height:26, borderRadius:8, background:`${s.color}18`, border:`1px solid ${s.color}30`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:800, color:s.color, flexShrink:0 }}>
                                                {i+1}
                                            </div>
                                            <div style={{ flex:1, minWidth:0 }}>
                                                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6 }}>
                                                    <div>
                                                        <span style={{ fontSize:13, fontWeight:700, color:C.text }}>{s.label}</span>
                                                        <span style={{ fontSize:10, color:C.muted, marginLeft:8 }}>{s.desc}</span>
                                                    </div>
                                                    <span style={{ fontSize:15, fontWeight:900, color:s.color, flexShrink:0, fontVariantNumeric:"tabular-nums", marginLeft:12 }}>{s.value}</span>
                                                </div>
                                                <SoftBar pct={pct} color={s.color} h={6} />
                                            </div>
                                        </div>
                                    </div>
                                );
                            });
                        })()}
                    </FieldCard>

                    <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"1fr 1fr", gap:14 }}>
                        {[
                            { label:"Shortlisted", count:eligible.length, color:C.green,   list:eligible, icon:"✓" },
                            { label:"Rejected",    count:rejected.length,  color:"#ef4444", list:rejected, icon:"✗" },
                        ].map(({ label, count, color, list, icon }) => (
                            <FieldCard key={label} label={label} dot={color}>
                                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
                                    <span style={{ fontSize:13, fontWeight:700, color }}>{count} candidates</span>
                                </div>
                                {/* Scrollable list — no "+N more" text */}
                                <div style={{ display:"flex", flexDirection:"column", gap:5, maxHeight:240, overflowY:"auto", paddingRight:2 }}>
                                    {list.map(c => (
                                        <div key={c.id} style={{ display:"flex", alignItems:"center", gap:9, padding:"7px 10px", borderRadius:9, background:`${color}05`, border:`1px solid ${color}10`, flexShrink:0 }}>
                                            <span style={{ fontSize:10, color, fontWeight:800, flexShrink:0 }}>{icon}</span>
                                            <span style={{ fontSize:12, color:C.text, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.name}</span>
                                            <span style={{ fontSize:11, fontWeight:700, color, flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{Math.round((c.finalScore||0)*100)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </FieldCard>
                        ))}
                    </div>
                </div>
            )}

            {/* ═══ TALENT ══════════════════════════════════════════════════════ */}
            {tab === "talent" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>

                    <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"repeat(3,1fr)", gap:10 }}>
                        {[
                            { label:"Avg Skill Match",    value:`${avgSkill}%`, color:C.blue,  sub:"skill extraction + alias matching" },
                            { label:"Avg Semantic Score", value:`${avgSem}%`,   color:C.teal,  sub:"cosine similarity vs job description" },
                            { label:"Avg Final Score",    value:`${avgFinal}%`, color:C.amber, sub:"weighted avg — shortlisted only" },
                        ].map(p => <StatPill key={p.label} {...p} />)}
                    </div>

                    <FieldCard label="Talent Quadrant Map" dot={C.teal}>
                        <p style={{ fontSize:12, color:C.cardText, marginBottom:14, lineHeight:1.6 }}>
                            Candidates split by skill coverage vs semantic alignment. <strong style={{color:C.text}}>Ideal Fit</strong> → strong on both → prioritise for interview.
                        </p>
                        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                            {[
                                { key:"tr", label:"Ideal Fit",  sub:"High skill · High semantic",  color:C.green,   list:quads.tr, cta:"INTERVIEW NOW" },
                                { key:"br", label:"Skilled",    sub:"High skill · Low semantic",   color:C.blue,    list:quads.br, cta:"CONSIDER"      },
                                { key:"tl", label:"Role-Aware", sub:"Low skill · High semantic",   color:C.amber,   list:quads.tl, cta:"REVIEW"        },
                                { key:"bl", label:"Weak Match", sub:"Low skill · Low semantic",    color:"#ef4444", list:quads.bl, cta:"PASS"          },
                            ].map(({ key, label, sub, color, list, cta }) => (
                                <div key={key} style={{ padding:"18px 18px 16px", borderRadius:14, background:`${color}06`, border:`1px solid ${color}20`, position:"relative", overflow:"hidden" }}>
                                    <div style={{ position:"absolute", top:0, left:0, right:0, height:1, background:`linear-gradient(90deg,transparent,${color}55,transparent)` }} />
                                    {/* Header */}
                                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:12 }}>
                                        <div>
                                            <div style={{ fontSize:14, fontWeight:800, color, letterSpacing:"-.01em" }}>{label}</div>
                                            <div style={{ fontSize:10, color:C.muted, marginTop:2 }}>{sub}</div>
                                        </div>
                                        <div style={{ textAlign:"right" }}>
                                            <div style={{ fontSize:26, fontWeight:900, color, fontVariantNumeric:"tabular-nums", lineHeight:1 }}>{list.length}</div>
                                            <div style={{ fontSize:8, fontWeight:800, color, letterSpacing:".06em", marginTop:2, opacity:.8 }}>{cta}</div>
                                        </div>
                                    </div>
                                    {/* Candidate rows — scrollable, each with score */}
                                    <div style={{ display:"flex", flexDirection:"column", gap:5, maxHeight:160, overflowY:"auto", scrollbarWidth:"none" }}>
                                        {list.map(c => {
                                            const fs = Math.round((c.finalScore||0)*100);
                                            return (
                                                <div key={c.id} style={{ display:"flex", alignItems:"center", gap:8, padding:"6px 9px", borderRadius:8, background:`${color}08`, border:`1px solid ${color}14` }}>
                                                    <div style={{ width:22, height:22, borderRadius:6, background:`${color}18`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:8, fontWeight:800, color, flexShrink:0 }}>
                                                        {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                                    </div>
                                                    <span style={{ fontSize:11, color:C.text, fontWeight:600, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.name}</span>
                                                    <span style={{ fontSize:11, fontWeight:800, color, flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{fs}%</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </FieldCard>

                    <FieldCard label="Top 10 Talent Rankings" dot={C.blue}>
                        <div style={{ display:"flex", justifyContent:"flex-end", marginBottom:12 }}>
                            <button onClick={() => onNav && onNav("candidates")} style={{ fontSize:11, fontWeight:600, color:C.blue, background:`${C.blue}10`, border:`1px solid ${C.blue}25`, borderRadius:8, padding:"4px 12px", cursor:"pointer", fontFamily:"inherit", display:"flex", alignItems:"center", gap:5 }}>
                                View All Candidates →
                            </button>
                        </div>
                        <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                            {eligible.slice(0,10).map((c,i) => {
                                const fs = Math.round((c.finalScore||0)*100);
                                const sk = Math.round((c.skillScore||0)*100);
                                const se = Math.round((c.semanticScore||0)*100);
                                const rankColor = i===0?C.amber:i===1?"#94a3b8":i===2?"#cd7f32":C.blue;
                                const medals = ["🥇","🥈","🥉"];
                                return (
                                    <div key={c.id} style={{ display:"flex", alignItems:"center", gap:12, padding:"11px 14px", borderRadius:12, background: i<3?`${rankColor}07`:C.surface, border:`1px solid ${i<3?`${rankColor}20`:C.border}`, transition:"background .15s" }}
                                        onMouseEnter={e=>e.currentTarget.style.background=`${C.blue}07`}
                                        onMouseLeave={e=>e.currentTarget.style.background=i<3?`${rankColor}07`:C.surface}>
                                        {/* Rank */}
                                        <div style={{ width:28, textAlign:"center", flexShrink:0, fontSize: i<3?15:11, fontWeight:800, color:rankColor, lineHeight:1 }}>
                                            {i<3?medals[i]:i+1}
                                        </div>
                                        {/* Avatar + name */}
                                        <div style={{ width:30, height:30, borderRadius:8, background:`${rankColor}18`, border:`1px solid ${rankColor}28`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:9, fontWeight:800, color:rankColor, flexShrink:0 }}>
                                            {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                        </div>
                                        <div style={{ flex:1, minWidth:0 }}>
                                            <div style={{ fontSize:12, fontWeight:700, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.name}</div>
                                            <div style={{ fontSize:9, color:C.muted }}>sk {sk}% · se {se}%</div>
                                        </div>
                                        {/* Talent score bar + final % */}
                                        <div style={{ width:100, flexShrink:0 }}>
                                            <SoftBar pct={fs} color={rankColor} h={5} />
                                        </div>
                                        <div style={{ width:36, textAlign:"right", fontSize:13, fontWeight:900, color:rankColor, flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{fs}%</div>
                                    </div>
                                );
                            })}
                        </div>
                    </FieldCard>
                </div>
            )}

            {/* ═══ SKILLS ══════════════════════════════════════════════════════ */}
            {tab === "skills" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>
                    <div style={{ display:"grid", gridTemplateColumns:isMobile?"1fr":"1fr 1fr", gap:14, alignItems:"start" }}>

                        <FieldCard label="Skill Coverage" dot={C.green}>
                            <p style={{ fontSize:11, color:C.cardText, marginBottom:14, lineHeight:1.5 }}>How many candidates match each required skill.</p>
                            <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                                {coverageData.map(({ skill, matched, pct }) => {
                                    const col = pct>=70?C.green:pct>=40?C.blue:pct>=20?C.amber:"#ef4444";
                                    return (
                                        <div key={skill} style={{ display:"flex", alignItems:"center", gap:10 }}>
                                            <div style={{ width:105, fontSize:11, fontWeight:500, color:C.text, textAlign:"right", flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{skill}</div>
                                            <div style={{ flex:1 }}><SoftBar pct={pct} color={col} h={6} /></div>
                                            <div style={{ width:42, fontSize:11, fontWeight:700, color:col, textAlign:"right", flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{matched}<span style={{color:C.muted,fontWeight:400}}>/{results.length}</span></div>
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard>

                        <FieldCard label="Skill Gap Intel" dot="#ef4444" xtra={{ border:`1px solid rgba(239,68,68,.2)` }}>
                            <p style={{ fontSize:11, color:C.cardText, marginBottom:14, lineHeight:1.5 }}>Required skills most commonly missing from the talent pool.</p>
                            {gapData.length === 0 ? (
                                <div style={{ textAlign:"center", padding:"16px", color:C.green, fontSize:12, fontWeight:700 }}>✓ No critical skill gaps detected</div>
                            ) : (
                                <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                                    {gapData.map(({ skill, count, pct }) => {
                                        const col = pct>=70?"#ef4444":pct>=40?"#f97316":pct>=20?C.amber:C.teal;
                                        const tag = pct>=70?"RARE":pct>=40?"SCARCE":pct>=20?"LOW":"OK";
                                        return (
                                            <div key={skill} style={{ display:"flex", alignItems:"center", gap:10 }}>
                                                <div style={{ width:105, fontSize:11, fontWeight:500, color:C.text, textAlign:"right", flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{skill}</div>
                                                <div style={{ flex:1 }}><SoftBar pct={pct} color={col} h={6} /></div>
                                                <div style={{ width:24, fontSize:11, fontWeight:700, color:col, textAlign:"right", flexShrink:0, fontVariantNumeric:"tabular-nums" }}>{count}</div>
                                                <span style={{ fontSize:8, padding:"2px 5px", borderRadius:4, fontWeight:700, background:`${col}14`, color:col, border:`1px solid ${col}20`, flexShrink:0, fontFamily:"monospace", minWidth:36, textAlign:"center" }}>{tag}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </FieldCard>
                    </div>

                    {eligible[0] && (eligible[0].matched_skills||[]).length >= 3 && (
                        <FieldCard label={`Top Candidate Skill Radar — ${eligible[0].name}`} dot={C.blue}>
                            <div style={{ fontSize:11, color:C.muted, marginBottom:14 }}>
                                Hover over each point to see the skill score. Each axis represents a required skill matched by this candidate.
                            </div>
                            <ResponsiveContainer width="100%" height={300}>
                                <RadarChart data={(() => {
                                    const c0 = eligible[0];
                                    const matched = new Set((c0.matched_skills||[]).map(s=>s.toLowerCase()));
                                    const allSkills = [...new Set([...(c0.matched_skills||[]), ...(c0.missing_skills||[])])].slice(0,6);
                                    const semBase = Math.round((c0.semanticScore||0)*100);
                                    return allSkills.map(s => ({
                                        skill: s,
                                        score: matched.has(s.toLowerCase())
                                            ? Math.min(100, Math.round(semBase * 0.4 + Math.round((c0.skillScore||0)*100) * 0.6))
                                            : Math.min(35, Math.round(semBase * 0.35)),
                                        fullMark: 100,
                                    }));
                                })()} outerRadius="70%" margin={{top:16,right:28,bottom:16,left:28}}>
                                    <PolarGrid gridType="circle" stroke={C.border} />
                                    <PolarAngleAxis dataKey="skill" tick={{ fill:C.sub, fontSize:11, fontWeight:600 }} />
                                    <Tooltip
                                        contentStyle={{ background:"rgba(9,9,11,.95)", border:`1px solid ${C.border}`, borderRadius:10, fontSize:12, backdropFilter:"blur(12px)", boxShadow:"0 8px 24px rgba(0,0,0,.5)", color:C.text, padding:"8px 12px" }}
                                        formatter={(value, name) => [`${value}%`, "Match Score"]}
                                        labelStyle={{ color:C.blue, fontWeight:700, marginBottom:2 }}
                                    />
                                    <Radar dataKey="score" stroke={C.blue} strokeWidth={2.5} fill={C.blue} fillOpacity={0.2} dot={{ fill:C.blue, r:5, strokeWidth:2, stroke:"rgba(255,255,255,0.3)" }} activeDot={{ r:7, fill:C.blue, stroke:"#fff", strokeWidth:2 }} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </FieldCard>
                    )}
                </div>
            )}

            {/* ═══ DECISIONS ═══════════════════════════════════════════════════ */}
            {tab === "decisions" && (
                <div style={{ display:"flex", flexDirection:"column", gap: 14 }}>

                    {/* Hiring Recommendation — full priority layout like quadrant map */}
                    <FieldCard label="Hiring Recommendation" dot={C.green}>
                        <p style={{ fontSize:11, color:C.sub, marginBottom:16, lineHeight:1.6 }}>
                            Action-prioritised summary for the HR team. Review each group and proceed accordingly.
                        </p>
                        {(() => {
                            const groups = [
                                { label:"Interview Now",    list:eligible.filter(c=>Math.round((c.finalScore||0)*100)>=70), color:C.green,   action:"Strong match — schedule interview", priority:"HIGH PRIORITY", cap:null },
                                { label:"Secondary Review", list:borderline.slice(0,3),                                       color:C.amber,   action:`${borderline.length} borderline candidates — see full list below`, priority:"REVIEW", cap:borderline.length, capNav:true },
                                { label:"Archive",          list:rejected,                                                    color:"#ef4444", action:"Did not qualify — send rejection", priority:"ARCHIVE", cap:null },
                            ];
                            return (
                                <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                                    {groups.map(({ label, list, color, action, priority, cap, capNav }) => (
                                        <div key={label} style={{ borderRadius:14, border:`1px solid ${color}20`, overflow:"hidden", background:`${color}04` }}>
                                            {/* Group header */}
                                            <div style={{ display:"flex", alignItems:"center", gap:12, padding:"12px 16px", borderBottom: list.length ? `1px solid ${color}12` : "none", background:`${color}07` }}>
                                                <div style={{ position:"relative" }}>
                                                    <div style={{ width:8, height:8, borderRadius:"50%", background:color }} />
                                                    <div style={{ position:"absolute", inset:0, borderRadius:"50%", background:color, opacity:.35, transform:"scale(2.2)" }} />
                                                </div>
                                                <div style={{ flex:1 }}>
                                                    <div style={{ fontSize:13, fontWeight:800, color }}>{label}</div>
                                                    <div style={{ fontSize:10, color:C.muted, marginTop:1 }}>{action}</div>
                                                </div>
                                                <div style={{ display:"flex", alignItems:"center", gap:10, flexShrink:0 }}>
                                                    <span style={{ fontSize:9, padding:"3px 9px", borderRadius:20, fontWeight:800, background:`${color}14`, color, border:`1px solid ${color}28`, letterSpacing:".05em" }}>{priority}</span>
                                                    <span style={{ fontSize:20, fontWeight:900, color, fontVariantNumeric:"tabular-nums", lineHeight:1 }}>{list.length}</span>
                                                </div>
                                            </div>
                                            {/* Scrollable candidate rows */}
                                            {list.length > 0 && (
                                                <div style={{ display:"flex", flexDirection:"column", gap:0, maxHeight:180, overflowY:"auto", scrollbarWidth:"none" }}>
                                                    {list.map((c, idx) => {
                                                        const fs = Math.round((c.finalScore||0)*100);
                                                        const sk = Math.round((c.skillScore||0)*100);
                                                        const se = Math.round((c.semanticScore||0)*100);
                                                        return (
                                                            <div key={c.id} style={{ display:"flex", alignItems:"center", gap:12, padding:"9px 16px", borderBottom: idx < list.length-1 ? `1px solid ${color}08` : "none", transition:"background .12s" }}
                                                                onMouseEnter={e=>e.currentTarget.style.background=`${color}07`}
                                                                onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
                                                                <div style={{ width:26, height:26, borderRadius:7, background:`${color}14`, border:`1px solid ${color}22`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:8, fontWeight:800, color, flexShrink:0 }}>
                                                                    {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                                                </div>
                                                                <div style={{ flex:1, minWidth:0 }}>
                                                                    <div style={{ fontSize:12, fontWeight:600, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{c.name}</div>
                                                                    <div style={{ fontSize:9, color:C.muted, marginTop:1 }}>sk {sk}% · se {se}%</div>
                                                                </div>
                                                                <div style={{ width:70, flexShrink:0 }}>
                                                                    <SoftBar pct={fs} color={color} h={4} />
                                                                </div>
                                                                <span style={{ fontSize:12, fontWeight:800, color, flexShrink:0, fontVariantNumeric:"tabular-nums", minWidth:32, textAlign:"right" }}>{fs}%</span>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            )}
                                            {list.length === 0 && (
                                                <div style={{ padding:"12px 16px", fontSize:11, color:C.muted }}>None in this group.</div>
                                            )}
                                            {capNav && cap > 3 && (
                                                <div style={{ padding:"8px 16px", borderTop:`1px solid ${color}10` }}>
                                                    <button onClick={() => {
                                                        const el = document.getElementById("borderline-section");
                                                        if (el) el.scrollIntoView({ behavior:"smooth", block:"start" });
                                                    }} style={{ fontSize:11, fontWeight:600, color, background:`${color}0e`, border:`1px solid ${color}25`, borderRadius:8, padding:"4px 12px", cursor:"pointer", fontFamily:"inherit" }}>
                                                        See all {cap} borderline candidates ↓
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            );
                        })()}
                    </FieldCard>

                    {/* Borderline */}
                    {borderline.length > 0 && (
                        <div id="borderline-section"><FieldCard label="Borderline Review" dot={C.amber} xtra={{ border:`1px solid ${C.amber}22` }}>
                            <p style={{ fontSize:12, color:C.cardText, marginBottom:14, lineHeight:1.6 }}>
                                These candidates scored <strong style={{color:C.amber}}>50–65%</strong> — shortlisted but close to the cutoff. Recommend manual review before final decision.
                            </p>
                            <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
                                {borderline.map(c => {
                                    const fs=Math.round((c.finalScore||0)*100), sk=Math.round((c.skillScore||0)*100), se=Math.round((c.semanticScore||0)*100);
                                    return (
                                        <div key={c.id} style={{ display:"flex", alignItems:"center", gap:12, padding:"11px 14px", borderRadius:12, background:`${C.amber}05`, border:`1px solid ${C.amber}15` }}>
                                            <div style={{ width:30,height:30,borderRadius:8,background:`${C.amber}15`,border:`1px solid ${C.amber}25`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,fontWeight:800,color:C.amber,flexShrink:0 }}>
                                                {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                            </div>
                                            <div style={{ flex:1, minWidth:0 }}>
                                                <div style={{ fontSize:12, fontWeight:700, color:C.text }}>{c.name}</div>
                                                <div style={{ fontSize:10, color:C.muted }}>{c.email||"No email"}</div>
                                            </div>
                                            <div style={{ display:"flex", gap:5, flexShrink:0 }}>
                                                {[["SK",sk,C.blue],["SE",se,C.teal],["FS",fs,C.amber]].map(([l,v,col])=>(
                                                    <div key={l} style={{ textAlign:"center", padding:"4px 7px", borderRadius:7, background:`${col}09`, border:`1px solid ${col}18`, minWidth:36 }}>
                                                        <div style={{ fontSize:11, fontWeight:800, color:col, fontVariantNumeric:"tabular-nums" }}>{v}%</div>
                                                        <div style={{ fontSize:8, color:C.muted }}>{l}</div>
                                                    </div>
                                                ))}
                                            </div>
                                            <span style={{ fontSize:9, padding:"3px 8px", borderRadius:20, background:`${C.amber}14`, color:C.amber, border:`1px solid ${C.amber}30`, fontWeight:700, fontFamily:"monospace", flexShrink:0 }}>REVIEW</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </FieldCard></div>
                    )}

                    {/* Rejection log */}
                    {rejected.length > 0 && (
                        <FieldCard label="Rejection Log" dot="#ef4444" xtra={{ border:"1px solid rgba(239,68,68,.18)" }}>
                            <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                                {rejected.map((c,i)=>(
                                    <div key={c.id||i} style={{ display:"flex", alignItems:"center", gap:11, padding:"9px 12px", borderRadius:11, background:"rgba(239,68,68,.03)", border:"1px solid rgba(239,68,68,.08)" }}>
                                        <div style={{ width:28,height:28,borderRadius:7,background:"rgba(239,68,68,.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,fontWeight:800,color:"#ef4444",flexShrink:0 }}>
                                            {(c.name||"?").split(" ").map(n=>n[0]).join("").slice(0,2).toUpperCase()}
                                        </div>
                                        <div style={{ flex:1, minWidth:0 }}>
                                            <div style={{ fontSize:12, fontWeight:700, color:C.text }}>{c.name||"Unknown"}</div>
                                            <div style={{ fontSize:10, color:"#ef4444", marginTop:1, fontFamily:"monospace" }}>✗ {c.rejection_reason||"Did not meet criteria"}</div>
                                        </div>
                                        <div style={{ display:"flex", gap:5, flexShrink:0 }}>
                                            {[["SK",Math.round((c.skillScore||0)*100),C.blue],["SE",Math.round((c.semanticScore||0)*100),C.teal]].map(([l,v,col])=>(
                                                <div key={l} style={{ textAlign:"center", padding:"3px 7px", borderRadius:6, background:`${col}09`, border:`1px solid ${col}16` }}>
                                                    <div style={{ fontSize:11, fontWeight:800, color:col, fontVariantNumeric:"tabular-nums" }}>{v}%</div>
                                                    <div style={{ fontSize:8, color:C.muted }}>{l}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </FieldCard>
                    )}
                </div>
            )}

        </div>
    );
};

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
    const [dark, setDark] = useState(true);
    const [nav, setNav] = useState("upload");
    const [screeningConfig, setScreeningConfig] = useState(null);
    const [results, setResults] = useState([]);          // real API results only
    const [activeModel, setActiveModel] = useState("mpnet");
    const [profileOpen, setProfileOpen] = useState(false);
    // Profile is loaded from localStorage so it survives page reloads
    const [profile, setProfile] = useState(() => {
        try {
            const saved = localStorage.getItem("screeningProfile");
            return saved ? JSON.parse(saved) : { name: "", email: "", role: "", org: "" };
        } catch { return { name: "", email: "", role: "", org: "" }; }
    });
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [backendOnline, setBackendOnline] = useState(null); // null=checking, true=online, false=offline

    // Poll /api/health every 5 s to reflect real backend status
    useEffect(() => {
        const check = async () => {
            try {
                const ctrl = new AbortController();
                const t = setTimeout(() => ctrl.abort(), 2500);
                const res = await fetch(`${BASE}/api/health`, { signal: ctrl.signal });
                clearTimeout(t);
                setBackendOnline(res.ok);
            } catch { setBackendOnline(false); }
        };
        check();
        const id = setInterval(check, 5000);
        return () => clearInterval(id);
    }, []);

    // Persist profile to localStorage whenever it changes
    const handleSaveProfile = (newProfile) => {
        setProfile(newProfile);
        try { localStorage.setItem("screeningProfile", JSON.stringify(newProfile)); } catch (_) { }
    };

    // Update module-level theme ref on every render
    C = dark ? DARK : LIGHT;

    const width = useWindowWidth();
    const isMobile = width < 768;
    const isTablet = width < 1080;

    const initials = (profile.name || "U").split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase() || "U";

    const navItems = [
        { id: "dashboard", icon: LayoutDashboard, label: "Dashboard" },
        { id: "upload", icon: Upload, label: "Upload" },
        { id: "config", icon: Briefcase, label: "Job Config" },
        { id: "candidates", icon: Users, label: "Candidates" },
        { id: "analytics", icon: BarChart2, label: "Analytics" },
    ];

    const go = id => { setNav(id); setSidebarOpen(false); };

    // Allow analytics tab to navigate via custom event
    useEffect(() => {
        const handler = e => go(e.detail);
        document.addEventListener("navTo", handler);
        return () => document.removeEventListener("navTo", handler);
    }, []);

    const handleStartScreening = cfg => {
        setScreeningConfig(cfg);
        setNav("processing");
        setSidebarOpen(false);
    };

    const handleProcessingDone = apiResults => {
        const r = apiResults || [];
        setResults(r);
        // Only redirect to upload if nothing was processed at all.
        // If results exist but all are rejected, still go to dashboard.
        setNav(r.length > 0 ? "dashboard" : "upload");
    };

    // Switch model — updates state and tells Flask
    const handleModelChange = async key => {
        setActiveModel(key);
        try {
            await fetch(`${BASE}/api/model`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: key }),
            });
        } catch (_) { }
    };

    // Map nav keys to view components
    const viewMap = {
        dashboard: <DashboardView results={results} onNav={go} isMobile={isMobile} activeModel={activeModel} onModelChange={handleModelChange} />,
        upload: <UploadView onStartScreening={handleStartScreening} activeModel={activeModel} onModelChange={handleModelChange} isMobile={isMobile} backendOnline={backendOnline} />,
        processing: <ProcessingView config={screeningConfig} onDone={handleProcessingDone} />,
        config: <JobConfigView />,
        candidates: <CandidatesView results={results} onNav={go} isMobile={isMobile} />,
        analytics: <AnalyticsView results={results} isMobile={isMobile} onNav={go} activeModel={activeModel} />,
    };

    // Sidebar content (shared between desktop sticky + mobile drawer)
    const SidebarContent = () => (
        <>
            {/* Logo — same height as topbar */}
            <div style={{ padding: "0 8px 20px", borderBottom: `1px solid ${C.border}`, marginBottom: 11 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 9, background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 4px 12px ${C.blue}44`, flexShrink: 0 }}>
                        <Sparkles size={15} color="#fff" />
                    </div>
                    <div>
                        <div style={{ fontSize: 13, fontWeight: 800, color: C.text, lineHeight: 1.1 }}>ML-Based</div>
                        <div style={{ fontSize: 10, color: C.sub, marginTop: 1 }}>Screening System</div>
                    </div>
                </div>
                {/* Developer credit */}
                <div style={{ marginTop: 20, padding: "7px 9px", borderRadius: 8, background: `${C.blue}08`, border: `1px solid ${C.blue}18` }}>
                    <div style={{ fontSize: 9, color: C.muted, textTransform: "uppercase", letterSpacing: ".08em", marginBottom: 2 }}>Developed by</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: C.blue }}>Madhan Kumar</div>
                    <div style={{ fontSize: 9, color: C.sub }}>B.Sc CS · Final Year Project</div>
                </div>
            </div>

            {/* Nav items */}
            <nav style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
                {navItems.map(({ id, icon: Icon, label }) => {
                    const active = nav === id;
                    return (
                        <button key={id} onClick={() => go(id)} style={{
                            width: "100%", display: "flex", alignItems: "center", gap: 10,
                            padding: "9px 10px", borderRadius: 9, border: "none",
                            background: active ? `${C.blue}14` : "transparent",
                            color: active ? C.blue : C.sub,
                            cursor: "pointer", fontSize: 13, fontWeight: active ? 700 : 500,
                            transition: "all .13s", fontFamily: "inherit",
                            borderLeft: `2px solid ${active ? C.blue : "transparent"}`,
                        }}
                            onMouseEnter={e => { if (!active) e.currentTarget.style.color = C.text; }}
                            onMouseLeave={e => { if (!active) e.currentTarget.style.color = C.sub; }}
                        >
                            <Icon size={14} /> {label}
                        </button>
                    );
                })}
            </nav>

            {/* Pipeline status — reflects real /api/health poll */}
            {(() => {
                const isOff = backendOnline === false;
                const isChecking = backendOnline === null;
                const col = isOff ? "#ef4444" : isChecking ? C.muted : C.green;
                const bg = isOff ? "rgba(239,68,68,.08)" : isChecking ? `${C.muted}08` : `${C.green}08`;
                const bd = isOff ? "rgba(239,68,68,.20)" : isChecking ? `${C.muted}20` : `${C.green}20`;
                const label = isOff ? "Pipeline Offline"
                    : isChecking ? "Checking\u2026"
                    : results.length > 0 ? `${results.length} candidates ranked`
                    : "Pipeline Ready";
                return (
                    <div style={{ padding: "8px 10px", borderRadius: 8, background: bg, border: `1px solid ${bd}`, fontSize: 11, color: col, display: "flex", alignItems: "center", gap: 7 }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: col, flexShrink: 0, animation: (!isOff && !isChecking) ? "pulse 2s infinite" : "none" }} />
                        {label}
                    </div>
                );
            })()}
        </>
    );

    return (
        <>
            {/* ── Global CSS ─────────────────────────────────────────────── */}
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');
        *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
        html { font-size:16px; }
        body { background:${C.bg}; font-family:'Plus Jakarta Sans',sans-serif; color:${C.text}; -webkit-font-smoothing:antialiased; font-weight:500; }
        ::-webkit-scrollbar { width:4px; height:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:${C.scrollThumb}; border-radius:2px; }

        /* Animations */
        @keyframes fadeIn      { from{opacity:0}                to{opacity:1} }
        @keyframes fadeSlideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
        @keyframes slideInRight{ from{transform:translateX(100%)} to{transform:none} }
        @keyframes slideInLeft { from{transform:translateX(-100%)} to{transform:none} }
        @keyframes spin        { to{transform:rotate(360deg)} }
        @keyframes pulse       { 0%,100%{opacity:1} 50%{opacity:.35} }

        /* Profile modal — spring pop in, GPU accelerated */
        @keyframes modalIn {
          from { opacity:0; transform:translate3d(-50%,-48%,0) scale(.97) }
          to   { opacity:1; transform:translate3d(-50%,-50%,0) scale(1) }
        }

        /* Range sliders */
        input[type=range] { -webkit-appearance:none; appearance:none; height:4px; background:${C.inputBg}; border-radius:2px; width:100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; border-radius:50%; background:currentColor; cursor:pointer; }
        textarea { color-scheme:${dark ? "dark" : "light"}; }
        button,input,textarea,select { font-family:'Plus Jakarta Sans',sans-serif; }
      `}</style>

            {/* ── Profile modal ──────────────────────────────────────────── */}
            {profileOpen && <ProfileModal profile={profile} onSave={handleSaveProfile} onClose={() => setProfileOpen(false)} />}

            {/* ── Mobile sidebar drawer ──────────────────────────────────── */}
            {isMobile && sidebarOpen && (
                <>
                    <div onClick={() => setSidebarOpen(false)} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", backdropFilter: "blur(4px)", zIndex: 300 }} />
                    <div style={{ position: "fixed", left: 0, top: 0, bottom: 0, width: 210, background: C.sidebarBg, borderRight: `1px solid ${C.border}`, zIndex: 310, padding: "20px 11px", display: "flex", flexDirection: "column", animation: "slideInLeft .24s ease" }}>
                        <SidebarContent />
                    </div>
                </>
            )}

            {/* ── Layout ─────────────────────────────────────────────────── */}
            <div style={{ display: "flex", minHeight: "100vh", background: C.bg, color: C.text }}>

                {/* Desktop sidebar */}
                {!isMobile && (
                    <aside style={{ width: isTablet ? 200 : 218, flexShrink: 0, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", padding: "20px 11px", position: "sticky", top: 0, height: "100vh", background: C.sidebarBg }}>
                        <SidebarContent />
                    </aside>
                )}

                {/* Main content area */}
                <main style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto", maxHeight: "100vh", minWidth: 0 }}>

                    {/* Topbar */}
                    <div style={{ padding: isMobile ? "11px 14px" : "14px 24px", borderBottom: `1px solid ${C.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", position: "sticky", top: 0, background: C.topbarBg, zIndex: 50, backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", gap: 8, minHeight: 64 }}>

                        {/* Left: hamburger (mobile) + page title */}
                        <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
                            {isMobile && (
                                <button onClick={() => setSidebarOpen(true)} style={{ width: 30, height: 30, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, flexShrink: 0 }}>
                                    <Menu size={14} />
                                </button>
                            )}
                            <div style={{ minWidth: 0 }}>
                                <div style={{ fontSize: 10, color: C.muted, fontWeight: 700, textTransform: "uppercase", letterSpacing: ".1em", whiteSpace: "nowrap" }}>
                                    {navItems.find(n => n.id === nav)?.label || "Processing"}
                                </div>
                                {!isMobile && (
                                    <div style={{ fontSize: 11, color: C.sub, marginTop: 1 }}>
                                        {new Date().toLocaleDateString("en-IN", { weekday: "long", month: "long", day: "numeric" })}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Right: model selector, theme toggle, profile */}
                        <div style={{ display: "flex", alignItems: "center", gap: 7, flexShrink: 0 }}>
                            <ModelDropdown activeModel={activeModel} onChange={handleModelChange} />

                            <button onClick={() => setDark(d => !d)} title="Toggle theme"
                                style={{ width: 30, height: 30, borderRadius: 7, border: `1px solid ${C.border}`, background: C.inputBg, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: C.sub, transition: "color .15s" }}
                                onMouseEnter={e => e.currentTarget.style.color = C.text}
                                onMouseLeave={e => e.currentTarget.style.color = C.sub}>
                                {dark ? <Sun size={13} /> : <Moon size={13} />}
                            </button>

                            <button onClick={() => setProfileOpen(true)} title={profile.name || "Edit Profile"}
                                style={{ width: 30, height: 30, borderRadius: 7, position: "relative", background: `linear-gradient(135deg,${C.blue},#a78bfa)`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, color: "#fff", border: "none", cursor: "pointer", boxShadow: `0 2px 8px ${C.blue}44` }}>
                                {initials}
                                <div style={{ position: "absolute", bottom: -1, right: -1, width: 9, height: 9, borderRadius: "50%", background: C.blue, border: `2px solid ${C.topbarBg}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                    <Edit3 size={4} color="#fff" />
                                </div>
                            </button>
                        </div>
                    </div>

                    {/* Page body */}
                    <div style={{ flex: 1, padding: isMobile ? "16px 13px" : "22px 26px", animation: "fadeIn .18s ease" }} key={nav}>
                        {viewMap[nav]}
                    </div>
                </main>
            </div>
        </>
    );
}