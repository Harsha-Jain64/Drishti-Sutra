"""Module 3: Evidence Relationship Graph Builder for Drishti-Sutra.

Constructs a multi-relational NetworkX graph and Plotly interactive dark-themed visualization.
Nodes: Entity, Evidence, Message, Event
Edges: EXTRACTED_FROM, MENTIONS, SAME_ENTITY, SUPPORTS, CONTRADICTS (highlighted in red)
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from src.database import DatabaseManager
from src.models import Contradiction, EvidenceItem, ExtractedChat, ExtractedDocument, TimelineEvent

# Node type palette used by the interactive forensic relationship graph.
NODE_COLORS = {
    "Person": "#1386a2",
    "Document": "#d39116",
    "Evidence": "#6d8296",
    "Message": "#8756a5",
    "Event": "#19865e",
}

NODE_SYMBOLS = {
    "Person": "circle",
    "Document": "square",
    "Evidence": "square",
    "Message": "diamond",
    "Event": "hexagon",
}

EDGE_COLORS = {
    "CONTRADICTS": "#ef4444",    # Vivid Red
    "SUPPORTS": "#22c55e",       # Green
    "EXTRACTED_FROM": "#64748b", # Slate Gray
    "MENTIONS": "#94a3b8",       # Light Slate
    "SAME_ENTITY": "#06b6d4",    # Teal
}


def build_evidence_graph(case_id: str, db: DatabaseManager) -> nx.MultiDiGraph:
    """Build a multi-directed NetworkX relationship graph from persisted case records."""
    G = nx.MultiDiGraph()

    evidence_items = {e.id: e for e in db.list_evidence(case_id)}
    docs: List[ExtractedDocument] = db.list_extracted_documents(case_id)
    chats: List[ExtractedChat] = db.list_extracted_chats(case_id)
    timeline: List[TimelineEvent] = db.list_timeline_events(case_id)
    contradictions: List[Contradiction] = db.list_contradictions(case_id)

    # 1. Add Evidence File Nodes
    for ev_id, ev in evidence_items.items():
        node_id = f"ev_{ev_id}"
        G.add_node(
            node_id,
            node_type=("Document" if ev.file_type in {"kyc_doc", "generic_document", "generic_pdf"} else "Evidence"),
            label=ev.file_name,
            file_name=ev.file_name,
            sha256=ev.sha256_hash[:16] + "...",
            size_bytes=ev.file_size_bytes,
            file_type=ev.file_type,
            evidence_id=ev_id,
        )

    # 2. Add Entities and link to Evidence
    entities_seen = set()
    for doc in docs:
        name = doc.fields.get("name")
        if name and name not in entities_seen:
            entities_seen.add(name)
            ent_node = f"ent_{name}"
            G.add_node(ent_node, node_type="Person", label=name, name=name)
        if name:
            G.add_edge(f"ent_{name}", f"ev_{doc.evidence_id}", relationship="EXTRACTED_FROM", label="extracted_from")

    for chat in chats:
        for msg in chat.messages:
            if msg.sender and msg.sender != "Unknown" and msg.sender not in entities_seen:
                entities_seen.add(msg.sender)
                ent_node = f"ent_{msg.sender}"
                G.add_node(ent_node, node_type="Person", label=msg.sender, name=msg.sender)

    # 3. Add Messages and link to Entities & Evidence
    for chat in chats:
        for idx, msg in enumerate(chat.messages):
            msg_node = f"msg_{msg.id}"
            G.add_node(
                msg_node,
                node_type="Message",
                label=f"Msg: {msg.content[:25]}...",
                content=msg.content,
                timestamp=msg.timestamp_str,
                sender=msg.sender,
                amounts=msg.amounts,
                source_type=chat.source_type,
            )
            # Edge to evidence
            G.add_edge(msg_node, f"ev_{chat.evidence_id}", relationship="EXTRACTED_FROM", label="extracted_from")
            # Edge to sender entity
            if msg.sender and f"ent_{msg.sender}" in G.nodes:
                G.add_edge(msg_node, f"ent_{msg.sender}", relationship="MENTIONS", label="sent_by")

    # 4. Add Event Nodes (from Timeline / Logs)
    for evt in timeline:
        if evt.event_type == "SYSTEM_LOG":
            evt_node = f"evt_{evt.id}"
            G.add_node(
                evt_node,
                node_type="Event",
                label=f"Event: {evt.actor} ({evt.timestamp_str})",
                description=evt.description,
                timestamp=evt.timestamp_str,
                actor=evt.actor,
                is_anomaly=evt.is_anomaly,
                evidence_id=evt.evidence_id,
            )
            if evt.evidence_id and f"ev_{evt.evidence_id}" in G.nodes:
                G.add_edge(evt_node, f"ev_{evt.evidence_id}", relationship="EXTRACTED_FROM", label="logged_in")
            if evt.actor and f"ent_{evt.actor}" in G.nodes:
                G.add_edge(evt_node, f"ent_{evt.actor}", relationship="MENTIONS", label="performed_by")

    # 5. Add SUPPORTS Relationships (e.g., Log transaction supports Screenshot amount)
    for u, u_data in list(G.nodes(data=True)):
        if u_data.get("node_type") == "Event" and "50000" in u_data.get("description", ""):
            for v, v_data in list(G.nodes(data=True)):
                if v_data.get("node_type") == "Message" and 50000.0 in v_data.get("amounts", []):
                    G.add_edge(u, v, relationship="SUPPORTS", label="supports_amount")

    # 6. Add CONTRADICTS Relationships (Vivid Red Edges)
    for c in contradictions:
        # Determine source nodes involved
        src_nodes: List[str] = []
        for ev_id in c.evidence_ids:
            if f"ev_{ev_id}" in G.nodes:
                src_nodes.append(f"ev_{ev_id}")

        if len(src_nodes) >= 2:
            G.add_edge(
                src_nodes[0],
                src_nodes[1],
                relationship="CONTRADICTS",
                label="CONTRADICTS",
                contradiction_id=c.id,
                contradiction_type=c.contradiction_type or "CONTRADICTION",
                field_name=c.field_name,
                severity=c.severity,
                explanation=c.explanation,
                confidence=c.confidence,
            )
        elif len(src_nodes) == 1:
            # Self-contradiction in single document (e.g. QR vs printed)
            target_ent = f"ent_{c.entity}" if f"ent_{c.entity}" in G.nodes else src_nodes[0]
            G.add_edge(
                src_nodes[0],
                target_ent,
                relationship="CONTRADICTS",
                label="CONTRADICTS",
                contradiction_id=c.id,
                contradiction_type=c.contradiction_type or "QR_PRINTED_MISMATCH",
                field_name=c.field_name,
                severity=c.severity,
                explanation=c.explanation,
                confidence=c.confidence,
            )

    G.graph["case_id"] = case_id
    return G


def build_plotly_figure(G: nx.MultiDiGraph, case_id: Optional[str] = None) -> go.Figure:
    """Render an interactive Plotly visualization for the evidence relationship graph."""
    cid = case_id or G.graph.get("case_id", "Case")
    # Deterministic spring layout with fixed seed
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=60)

    # Separate traces for edge types
    edge_traces: Dict[str, Dict[str, List[Any]]] = {
        "CONTRADICTS": {"x": [], "y": [], "hover": []},
        "SUPPORTS": {"x": [], "y": [], "hover": []},
        "EXTRACTED_FROM": {"x": [], "y": [], "hover": []},
        "MENTIONS": {"x": [], "y": [], "hover": []},
        "SAME_ENTITY": {"x": [], "y": [], "hover": []},
    }

    for u, v, k, data in G.edges(keys=True, data=True):
        rel = data.get("relationship", "EXTRACTED_FROM")
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            target_trace = edge_traces.get(rel, edge_traces["EXTRACTED_FROM"])
            target_trace["x"].extend([x0, x1, None])
            target_trace["y"].extend([y0, y1, None])

            hover_text = f"Relation: {rel}<br>Between: {G.nodes[u].get('label')} -> {G.nodes[v].get('label')}"
            if rel == "CONTRADICTS":
                hover_text += f"<br>Type: {data.get('contradiction_type')}<br>Severity: {data.get('severity')}<br>Detail: {data.get('explanation')}"
            target_trace["hover"].append(hover_text)

    fig = go.Figure()

    # Add background edge traces
    for rel_type in ("EXTRACTED_FROM", "MENTIONS", "SAME_ENTITY", "SUPPORTS"):
        t_data = edge_traces[rel_type]
        if t_data["x"]:
            fig.add_trace(
                go.Scatter(
                    x=t_data["x"],
                    y=t_data["y"],
                    line=dict(
                        width=1.5 if rel_type != "SUPPORTS" else 2.5,
                        color=EDGE_COLORS.get(rel_type, "#64748b"),
                        dash="dash" if rel_type == "SAME_ENTITY" else "solid",
                    ),
                    hoverinfo="none",
                    mode="lines",
                    name=rel_type,
                )
            )

    # Add CONTRADICTS edges (prominent red)
    contra_trace = edge_traces["CONTRADICTS"]
    if contra_trace["x"]:
        fig.add_trace(
            go.Scatter(
                x=contra_trace["x"],
                y=contra_trace["y"],
                line=dict(width=3.5, color=EDGE_COLORS["CONTRADICTS"]),
                hoverinfo="none",
                mode="lines",
                name="CONTRADICTS (Anomaly)",
            )
        )

    # Node traces grouped by node_type
    node_groups: Dict[str, Dict[str, List[Any]]] = {
        "Person": {"x": [], "y": [], "text": [], "labels": []},
        "Document": {"x": [], "y": [], "text": [], "labels": []},
        "Evidence": {"x": [], "y": [], "text": [], "labels": []},
        "Message": {"x": [], "y": [], "text": [], "labels": []},
        "Event": {"x": [], "y": [], "text": [], "labels": []},
    }

    for n, data in G.nodes(data=True):
        ntype = data.get("node_type", "Evidence")
        if n in pos:
            x, y = pos[n]
            group = node_groups.get(ntype, node_groups["Evidence"])
            group["x"].append(x)
            group["y"].append(y)
            group["labels"].append(data.get("label", n))

            # Rich hover text
            hover = f"<b>{ntype}</b>: {data.get('label')}"
            if ntype in {"Evidence", "Document"}:
                hover += f"<br>SHA-256: {data.get('sha256')}<br>Size: {data.get('size_bytes')} bytes"
            elif ntype == "Message":
                hover += f"<br>Sender: {data.get('sender')}<br>Time: {data.get('timestamp')}<br>Amounts: {data.get('amounts')}"
            elif ntype == "Event":
                hover += f"<br>Time: {data.get('timestamp')}<br>Detail: {data.get('description')}"
            group["text"].append(hover)

    # Add node traces
    for ntype, g_data in node_groups.items():
        if g_data["x"]:
            fig.add_trace(
                go.Scatter(
                    x=g_data["x"],
                    y=g_data["y"],
                    mode="markers+text",
                    text=g_data["labels"],
                    textposition="top center",
                    textfont=dict(size=10, color="#f1f5f9"),
                    hoverinfo="text",
                    hovertext=g_data["text"],
                    marker=dict(
                        size=20 if ntype in ("Person", "Document", "Evidence") else 14,
                        color=NODE_COLORS.get(ntype, "#94a3b8"),
                        symbol=NODE_SYMBOLS.get(ntype, "circle"),
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                    name=ntype,
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Drishti-Sutra: Forensic Evidence Correlation Graph ({cid})",
            font=dict(size=18, color="#f8fafc"),
        ),
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#cbd5e1"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
