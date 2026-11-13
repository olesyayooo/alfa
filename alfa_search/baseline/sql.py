"""BM25F ranking SQL; catalog retrieval is shared by both searches."""

# Parameters: terms, weights, b, k1, limit. Aggregate normalized TF across fields
# before applying saturation (BM25F, not a sum of independent BM25 scores).
SEARCH_SQL = '''
WITH term_tf AS (
    SELECT p.material_type, p.material_id, t.term_id, t.document_frequency,
           s.document_count,
           SUM(($2::double precision[])[p.field] * p.tf /
               (1 - ($3::double precision[])[p.field] + ($3::double precision[])[p.field]
                * m.field_lengths[p.field] / NULLIF(s.avg_lengths[p.field], 0))) AS weighted_tf
    FROM search.terms t
    JOIN search.postings p USING (term_id)
    JOIN search.materials m USING (material_type, material_id)
    CROSS JOIN search.index_stats s
    WHERE t.term = ANY($1::text[])
    GROUP BY p.material_type,p.material_id,t.term_id,t.document_frequency,s.document_count
), ranked AS (
    SELECT material_type, material_id,
           SUM(ln(1 + (document_count - document_frequency + 0.5) /
                        (document_frequency + 0.5)) *
               (($4::double precision + 1) * weighted_tf) /
               ($4::double precision + weighted_tf)) AS score
    FROM term_tf GROUP BY material_type,material_id
)
SELECT material_id,material_type,score FROM ranked WHERE score > 0
ORDER BY score DESC,material_type,material_id LIMIT $5
'''
