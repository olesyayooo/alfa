"""SQL and verified source-table mappings. No user-supplied identifiers."""

PUBLISHED = "61f12dbc-3cd4-4c14-b64b-1709548b904e"
SOURCES = (
    ("Courses", "course", "8b440b1a-00c3-423a-9bae-3376475eb203", True),
    ("Articles", "article", "8ede4179-0faf-407d-a4fc-1d94f9a63089", True),
    ("Documents", "document", "da1ad865-8605-4ab1-9f6c-4489a26fbc35", False),
    ("Webinars", "webinar", "ece4acf3-5dd3-47b9-9f08-99c0d6203cc7", True),
    ("Podcasts", "podcast", "142c23e2-bf71-4a73-b12e-6b1b5e4d47cc", True),
)


def catalog_sql(exclude_closed: bool, public_articles_only: bool) -> str:
    branches = []
    for table, kind, entity, has_closed in SOURCES:
        conditions = []
        if exclude_closed and has_closed:
            conditions.append('NOT m."IsClosed"')
        if public_articles_only and kind == "article":
            conditions.append('m."IsPublic"')
        extra = "".join(" AND " + c for c in conditions)
        branches.append(f'''SELECT m."Id" AS id, '{kind}'::text AS type,
            m."Title"::text AS title, m."Description"::text AS description,
            cs."Id" AS settings_id
            FROM public."{table}" m JOIN public."ContentSettings" cs
              ON cs."EntityId"=m."Id" AND cs."EntityTypeId"='{entity}'::uuid
            WHERE NOT m."IsDeleted" AND NOT cs."IsDeleted"
              AND m."StatusId"=$1::uuid AND cs."StatusId"=$1::uuid {extra}''')
    aggregates = []
    for table, singular, alias in (("Topics", "Topic", "topics"),
                                   ("Tags", "Tag", "tags"),
                                   ("Industries", "Industry", "industries")):
        aggregates.append(f'''COALESCE((SELECT array_agg(DISTINCT d."Title"::text ORDER BY d."Title"::text)
            FROM public."ContentSettings{table}" l JOIN public."{table}" d ON d."Id"=l."{singular}Id"
            WHERE l."ContentSettingsId"=m.settings_id AND NOT d."IsDeleted"), ARRAY[]::text[]) AS {alias}''')
    return "WITH materials AS (" + " UNION ALL ".join(branches) + ") SELECT id,type,title,description," + ",".join(aggregates) + " FROM materials m ORDER BY type,id"


