"""Startup tracker agent (LangGraph). Maps over watchlist companies.

Graph contract: resolve_board → fetch_signals → snapshot → diff → branch
(meaningful change?) → synthesize_dossier → score_trending → rank_and_route.
Only resolve_board (JAR-54) is implemented so far.
"""
