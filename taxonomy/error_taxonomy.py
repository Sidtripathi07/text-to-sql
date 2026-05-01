"""
taxonomy/error_taxonomy.py
Full 31-subtype SQL error taxonomy used by the Correction Plan Agent.
Derived from Shen et al. (2025) and extended for LLM-friendly error codes.
"""

# ── Taxonomy as structured dict ───────────────────────────────────────────────
TAXONOMY: dict[str, dict] = {
    "syntax": {
        "description": "Syntactic errors that make the SQL unparseable or invalid",
        "subtypes": {
            "sql_syntax_error"  : "General SQL syntax violation — missing keyword, bracket, comma.",
            "invalid_alias"     : "Alias defined incorrectly or referenced before assignment.",
        },
    },
    "schema_link": {
        "description": "Errors in referencing tables or columns",
        "subtypes": {
            "table_missing"         : "A required table was not included in FROM/JOIN.",
            "col_missing"           : "A required column was omitted from SELECT or WHERE.",
            "ambiguous_col"         : "Column name used without table qualifier causing ambiguity.",
            "incorrect_foreign_key" : "Wrong column used as join key between tables.",
        },
    },
    "join": {
        "description": "Errors in joining tables",
        "subtypes": {
            "join_missing"    : "Two tables used together but no JOIN condition specified.",
            "join_wrong_type" : "INNER used where LEFT/RIGHT or vice-versa was needed.",
            "extra_table"     : "A table was joined but contributes nothing to the result.",
            "incorrect_col"   : "Wrong column referenced in ON clause of a JOIN.",
        },
    },
    "filter": {
        "description": "Errors in WHERE / HAVING filter conditions",
        "subtypes": {
            "where_missing"          : "A required WHERE filter was completely omitted.",
            "condition_wrong_col"    : "Filter applied to wrong column.",
            "condition_type_mismatch": "Comparing string to integer or date without cast.",
        },
    },
    "aggregation": {
        "description": "Errors in GROUP BY / aggregation functions",
        "subtypes": {
            "agg_no_groupby"       : "Aggregate function used without GROUP BY.",
            "groupby_missing_col"  : "Not all non-aggregate SELECT columns are in GROUP BY.",
            "having_without_groupby": "HAVING clause used without a GROUP BY.",
            "having_incorrect"     : "HAVING condition references wrong column or aggregation.",
            "having_vs_where"      : "Filter that belongs in WHERE was placed in HAVING.",
        },
    },
    "value": {
        "description": "Errors related to literal values",
        "subtypes": {
            "hardcoded_value"  : "A literal value used where a column reference was needed.",
            "value_format_wrong": "Date/string formatted incorrectly for the DB engine.",
        },
    },
    "subquery": {
        "description": "Errors in nested / correlated subqueries",
        "subtypes": {
            "unused_subquery"          : "Subquery written but result never used in outer query.",
            "subquery_missing"         : "A needed subquery was not written; flat query used instead.",
            "subquery_correlation_error": "Correlated subquery references wrong outer alias.",
        },
    },
    "set_operations": {
        "description": "Errors in UNION / INTERSECT / EXCEPT",
        "subtypes": {
            "union_missing"    : "UNION needed to combine result sets but not used.",
            "intersect_missing": "INTERSECT needed but not used.",
            "except_missing"   : "EXCEPT / MINUS needed but not used.",
        },
    },
    "other": {
        "description": "Structural oversights",
        "subtypes": {
            "order_by_missing"     : "Result requires ordering but ORDER BY omitted.",
            "limit_missing"        : "Top-N query but LIMIT clause omitted.",
            "duplicate_select"     : "Same column selected multiple times redundantly.",
            "unsupported_function" : "Used a SQL function not supported by SQLite.",
            "extra_values_selected": "SELECT includes columns not asked for in the question.",
        },
    },
}


def get_compact_taxonomy_string() -> str:
    """
    Returns a token-efficient string representation of the taxonomy
    for injection into LLM prompts.
    """
    lines = ["SQL ERROR TAXONOMY (use exact codes):"]
    for category, info in TAXONOMY.items():
        lines.append(f"\n[{category.upper()}] {info['description']}")
        for code, desc in info["subtypes"].items():
            lines.append(f"  {code}: {desc}")
    return "\n".join(lines)


def get_all_codes() -> list[str]:
    """Return flat list of all 31 error codes."""
    codes = []
    for cat in TAXONOMY.values():
        codes.extend(cat["subtypes"].keys())
    return codes


COMPACT_TAXONOMY = get_compact_taxonomy_string()
ALL_ERROR_CODES  = get_all_codes()