"""Schémas Pandera des sources et du manifeste ; les lignes invalides sont rejetées et journalisées."""
try:
    import pandera.pandas as pa
except ImportError:  # anciennes versions de pandera
    import pandera as pa

HM_SCHEMA = pa.DataFrameSchema({
    "id": pa.Column(int, pa.Check.ge(0), unique=True, coerce=True),
    "img": pa.Column(str, pa.Check.str_matches(r"^img/.+\.(png|jpg|jpeg)$")),
    "text": pa.Column(str, nullable=True),
    "label": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=True, coerce=True),
    "orig_split": pa.Column(str, pa.Check.isin(["train", "dev", "test"])),
}, strict=False)

S140_SCHEMA = pa.DataFrameSchema({
    "target": pa.Column(int, pa.Check.isin([0, 4])),
    "ids": pa.Column(int, coerce=True),
    "text": pa.Column(str),
}, strict=False)

MANIFEST_SCHEMA = pa.DataFrameSchema({
    "sample_id": pa.Column(str, unique=True),
    "source": pa.Column(str, pa.Check.isin(["hateful_memes", "sentiment140", "reddit"])),
    "text": pa.Column(str),
    "text_clean": pa.Column(str),
    "img_path": pa.Column(str, nullable=True),
    "img_hash": pa.Column(str, nullable=True),
    "lang": pa.Column(str),
    "lang_conf": pa.Column(float, pa.Check.in_range(0, 1)),
    "label": pa.Column(str, pa.Check.isin(["hate", "non_hate", "negative", "positive", "unlabeled"])),
    "label_source": pa.Column(str),
    "split": pa.Column(str, pa.Check.isin(["train", "val", "test", "unlabeled", "external_test"])),
    "collected_at": pa.Column(str),
}, strict=False, coerce=True)

SchemaErrors = pa.errors.SchemaErrors


def validate_schema(df, schema, source, name_col, rl):
    """Valide df ; rejette (et journalise) les lignes invalides. Erreur structurelle : on arrête."""
    try:
        schema.validate(df, lazy=True)
        rl.event(step="schema", source=source, status="PASS", rows=len(df))
        return df
    except SchemaErrors as err:
        fc = err.failure_cases
        bad = fc["index"].dropna().astype(int).unique() if "index" in fc else []
        rl.event(step="schema", source=source, status="FAIL", n_failures=len(fc))
        print(fc.head(10))
        if len(bad) == 0:
            raise
        rl.reject_bulk(source, df.loc[bad, name_col], "schéma invalide")
        return df.drop(index=bad)
