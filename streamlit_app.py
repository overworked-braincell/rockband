from pathlib import Path
import contextlib
import io
import traceback
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="Ben's Rock Band: Machine Learning", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-1: #0b1220;
        --bg-2: #111827;
        --panel: #1f2937;
        --panel-soft: #111827;
        --text: #e5e7eb;
        --line: #374151;
        --accent-1: #f97316;
        --accent-2: #14b8a6;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, #111827 0%, #0b1220 58%, #0f172a 100%);
        color: var(--text);
    }
    h1, h2, h3, h4, h5, h6, p, span, label, li, div {
        color: var(--text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }
    .stButton > button {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: white;
        border: none;
        border-radius: 8px;
    }
    .hero {
        background: linear-gradient(100deg, rgba(249,115,22,0.2), rgba(20,184,166,0.2));
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }
    .hero h2 {
        margin: 0;
        color: #f9fafb;
    }
    .hero p {
        color: #d1d5db;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
            <h2>Ben's Rock Band: Performance Score Predictor</h2>
            <p>You were just hired as Ben's data scientist.</p>
            <p>Audit past gigs, model the next show's PERFORMANCE SCORE, and give Ben a data-backed set strategy before soundcheck.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    primary = Path("joined_dataset.csv")
    fallback = Path("joined_dataset - Copy.csv")
    data_path = primary if primary.exists() else fallback
    if not data_path.exists():
        raise FileNotFoundError("Could not find joined_dataset.csv or joined_dataset - Copy.csv.")

    df = pd.read_csv(data_path)
    if "PERFORMANCE_DATETIME" in df.columns:
        df["PERFORMANCE_DATETIME"] = pd.to_datetime(df["PERFORMANCE_DATETIME"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def build_performance_table(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    required_cols = [
        "PERFORMANCE_ID",
        "SCORE",
        "RATING",
        "SONG_NAME",
        "ARTIST",
        "RELEASE_YEAR",
        "SONG_DIFFICULTY_RATING",
        "SONG_FUN_RATING",
        "PERFORMANCE_DATETIME",
        "PLAYER_ID",
        "INSTRUMENT_ID",
        "PREFERRED_INSTRUMENT_ID",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    for col in ["SCORE", "RATING", "RELEASE_YEAR", "SONG_DIFFICULTY_RATING", "SONG_FUN_RATING"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    base_cols = [
        "PERFORMANCE_ID",
        "SCORE",
        "RATING",
        "SONG_NAME",
        "ARTIST",
        "RELEASE_YEAR",
        "SONG_DIFFICULTY_RATING",
        "SONG_FUN_RATING",
        "PERFORMANCE_DATETIME",
    ]

    perf_base = work[base_cols].drop_duplicates("PERFORMANCE_ID").set_index("PERFORMANCE_ID")
    team_agg = work.groupby("PERFORMANCE_ID").agg(
        TEAM_SIZE=("PLAYER_ID", "nunique"),
        INSTRUMENTS_COVERED=("INSTRUMENT_ID", "nunique"),
    )

    work["PREF_MATCH"] = (work["INSTRUMENT_ID"] == work["PREFERRED_INSTRUMENT_ID"]).astype(int)
    pref_ratio = work.groupby("PERFORMANCE_ID")["PREF_MATCH"].mean().rename("PREF_MATCH_RATIO")

    perf = perf_base.join(team_agg).join(pref_ratio).reset_index()
    perf["MONTH"] = perf["PERFORMANCE_DATETIME"].dt.month
    perf["DAY_OF_WEEK"] = perf["PERFORMANCE_DATETIME"].dt.dayofweek

    for col in perf.select_dtypes(include="number").columns:
        perf[col] = perf[col].fillna(perf[col].median())

    for col in ["SONG_NAME", "ARTIST"]:
        if perf[col].isna().any():
            perf[col] = perf[col].fillna(perf[col].mode().iloc[0])

    perf = perf.dropna(subset=["SCORE"]).copy()

    if remove_outliers:
        q1, q3 = perf["SCORE"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        perf = perf[(perf["SCORE"] >= lower) & (perf["SCORE"] <= upper)].copy()

    return perf


def get_feature_columns(perf: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    leakage_cols = {"PERFORMANCE_ID", "SCORE", "RATING", "PERFORMANCE_DATETIME"}
    feature_cols = [c for c in perf.columns if c not in leakage_cols]
    cat_cols = [c for c in ["SONG_NAME", "ARTIST"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


def build_pipeline(model_name: str, seed: int, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    if model_name == "LinearRegression":
        estimator = LinearRegression()
    elif model_name == "RandomForestRegressor":
        estimator = RandomForestRegressor(
            n_estimators=220,
            random_state=seed,
            max_features="sqrt",
            min_samples_leaf=1,
        )
    elif model_name == "KNeighborsRegressor":
        estimator = KNeighborsRegressor(n_neighbors=7)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("prep", preprocessor), ("model", estimator)], memory=None)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_holdout(
    perf: pd.DataFrame,
    model_names: list[str],
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, pd.DataFrame | pd.Series]]:
    feature_cols, cat_cols, num_cols = get_feature_columns(perf)
    x = perf[feature_cols]
    y = perf["SCORE"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}

    naive_pred_train = np.full(shape=len(y_train), fill_value=float(y_train.mean()))
    naive_pred_test = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    naive_train = regression_metrics(y_train, naive_pred_train)
    naive_test = regression_metrics(y_test, naive_pred_test)

    rows.append(
        {
            "Model": "NaiveMean",
            "Train_MAE": round(naive_train["MAE"], 2),
            "Train_RMSE": round(naive_train["RMSE"], 2),
            "Train_R2": round(naive_train["R2"], 4),
            "Test_MAE": round(naive_test["MAE"], 2),
            "Test_RMSE": round(naive_test["RMSE"], 2),
            "Test_R2": round(naive_test["R2"], 4),
        }
    )

    for name in model_names:
        model = build_pipeline(name, seed, cat_cols, num_cols)
        model.fit(x_train, y_train)
        trained_models[name] = model

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)
        train_m = regression_metrics(y_train, pred_train)
        test_m = regression_metrics(y_test, pred_test)

        rows.append(
            {
                "Model": name,
                "Train_MAE": round(train_m["MAE"], 2),
                "Train_RMSE": round(train_m["RMSE"], 2),
                "Train_R2": round(train_m["R2"], 4),
                "Test_MAE": round(test_m["MAE"], 2),
                "Test_RMSE": round(test_m["RMSE"], 2),
                "Test_R2": round(test_m["R2"], 4),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
    split_data: dict[str, pd.DataFrame | pd.Series] = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return metrics_df, trained_models, split_data


def run_prompt_code(code: str, df: pd.DataFrame, perf: pd.DataFrame) -> dict[str, object]:
    allowed_builtins = {
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "range": range,
        "sorted": sorted,
        "print": print,
        "float": float,
        "int": int,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
    }

    env: dict[str, object] = {
        "__builtins__": allowed_builtins,
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "perf": perf.copy(),
    }

    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, env, env)
    except Exception:
        return {
            "ok": False,
            "stdout": stdout_buffer.getvalue(),
            "result": None,
            "error": traceback.format_exc(),
        }

    return {
        "ok": True,
        "stdout": stdout_buffer.getvalue(),
        "result": env.get("result"),
        "error": "",
    }


def render_prompt_help(
    example_code: str,
    runnable_code: str,
    prompt_key: str,
    challenge_question: str,
    challenge_answer: str,
) -> None:
    answer_key = f"unlock_answer_{prompt_key}"
    current_answer = str(st.session_state.get(answer_key, "")).strip()

    with st.expander("Pseudo-code", expanded=False):
        st.markdown("### Pseudo-code guidance")
        st.markdown("Use this as a simple recipe. Read one line, write one matching Python line.")
        st.code(example_code.strip(), language="python")

    with st.expander("Runnable example (locked)", expanded=bool(current_answer)):
        st.markdown("### Challenge unlock")
        st.markdown("Answer the challenge question correctly to reveal the runnable reference code.")
        st.markdown(challenge_question)
        user_answer = st.text_input(
            "Enter your answer to unlock",
            key=answer_key,
        )

        if user_answer.strip().lower() == challenge_answer.strip().lower():
            st.success("Correct. Runnable example unlocked.")
            st.markdown("Use this as a starter template and edit it in your own style.")
            st.code(runnable_code.strip(), language="python")
        elif user_answer.strip():
            st.warning("Not quite. Try again.")


def render_code_runner(label: str, key: str, df: pd.DataFrame, perf: pd.DataFrame) -> None:
    code_key = f"prompt_code_{key}"
    if code_key not in st.session_state:
        st.session_state[code_key] = ""

    st.caption("Use `df` as your main input dataframe.")
    st.text_area(label, key=code_key, height=160)
    c1, c2 = st.columns(2)
    run_clicked = c1.button("Run code", key=f"run_{key}", type="primary")
    clear_clicked = c2.button("Clear", key=f"clear_{key}")

    if clear_clicked:
        st.session_state[code_key] = ""
        st.rerun()

    if run_clicked:
        result = run_prompt_code(st.session_state[code_key], df=df, perf=perf)

        if result["ok"]:
            st.success("Code ran successfully.")
        else:
            st.error("Code failed. See error details below.")

        stdout_text = str(result["stdout"]).strip()
        if stdout_text:
            st.markdown("Output")
            st.code(stdout_text, language="text")

        result_obj = result["result"]
        if isinstance(result_obj, pd.DataFrame):
            st.markdown("Result variable")
            st.dataframe(result_obj, use_container_width=True)
        elif result_obj is not None:
            st.markdown("Result variable")
            st.write(result_obj)

        error_text = str(result["error"]).strip()
        if error_text:
            st.code(error_text, language="text")


def render_team_results_store(
    remove_outliers: bool,
    test_size_pct: int,
    random_seed: int,
) -> None:
    st.subheader("Team Results Submission")
    st.caption("Save each team submission to a CSV so results can be reviewed later.")

    with st.form("team_results_form"):
        team_name = st.text_input("Team name")
        team_members = st.text_input("Team members (comma-separated)")
        submission_notes = st.text_area("Submission notes", height=80)
        save_submission = st.form_submit_button("Save team results", type="primary")

    if not save_submission:
        return

    if not team_name.strip():
        st.error("Please enter a team name before saving.")
        return

    metrics_df = st.session_state.holdout_metrics
    best_model = ""
    best_test_rmse = np.nan
    metrics_json = ""

    if metrics_df is not None and not metrics_df.empty:
        non_naive = metrics_df[metrics_df["Model"] != "NaiveMean"]
        if not non_naive.empty:
            best_row = non_naive.sort_values("Test_RMSE", ascending=True).iloc[0]
            best_model = str(best_row["Model"])
            best_test_rmse = float(best_row["Test_RMSE"])
        metrics_json = metrics_df.to_json(orient="records")

    result_row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "team_name": team_name.strip(),
        "team_members": team_members.strip(),
        "notes": submission_notes.strip(),
        "remove_outliers": bool(remove_outliers),
        "test_size_pct": int(test_size_pct),
        "random_seed": int(random_seed),
        "best_model": best_model,
        "best_test_rmse": best_test_rmse,
        "analysis_answer": str(st.session_state.get("analysis_answer", "")).strip(),
        "prompt1_code": str(st.session_state.get("prompt_code_p1", "")),
        "prompt2_code": str(st.session_state.get("prompt_code_p2_shape", "")),
        "prompt3_code": str(st.session_state.get("prompt_code_p3_stats", "")),
        "prompt3_outlier_code": str(st.session_state.get("prompt_code_p3_outliers", "")),
        "prompt4_code": str(st.session_state.get("prompt_code_p4_drop", "")),
        "metrics_json": json.dumps(json.loads(metrics_json)) if metrics_json else "",
    }

    output_path = Path("team_results.csv")
    new_row_df = pd.DataFrame([result_row])

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        save_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        save_df = new_row_df

    save_df.to_csv(output_path, index=False)
    st.success(f"Saved team results to {output_path.name}")
    st.dataframe(new_row_df, use_container_width=True)


if "holdout_metrics" not in st.session_state:
    st.session_state.holdout_metrics = None
if "holdout_models" not in st.session_state:
    st.session_state.holdout_models = {}
if "split_data" not in st.session_state:
    st.session_state.split_data = None


try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

remove_outliers = bool(st.session_state.get("model_remove_outliers", True))
test_size_pct = int(st.session_state.get("model_test_size_pct", 20))
random_seed = int(st.session_state.get("model_random_seed", 42))

try:
    perf = build_performance_table(df, remove_outliers=remove_outliers)
except Exception as exc:
    st.error(f"Unable to build performance-level table: {exc}")
    st.stop()

st.subheader("Story Mode: Regression Challenge Flow")
st.info(
    "Mission briefing: Ben hired you to stop guesswork before live shows. "
    "You will inspect gig history, build prediction models, and explain what lifts or hurts PERFORMANCE SCORE. "
    "Dataframe naming rule: use `df` for the source dataset."
)

with st.sidebar:
    st.markdown("### Pseudocode Translation Tips")
    st.markdown("- `Set X to ...` usually becomes `x = ...` in Python.")
    st.markdown("- `FOR each ...` usually becomes a `for` loop.")
    st.markdown("- `IF ...` usually becomes an `if` statement.")
    st.markdown("- Keep assigning your final output to a varaible so the app can display it.")
    st.markdown("- If you get stuck, write only Step 1 first, run it, then add the next step.")

st.markdown(
    """
Scene 1: Backstage intake - open Ben's gig history and preview rows.\n
Scene 2: Feature planning - define target and choose candidate predictors.\n
Scene 3: Crowd pulse check - profile statistics to understand score behavior.\n
Scene 4: Setlist cleanup - keep only useful columns for fair prediction.\n
Scene 5: Rehearsal lab - tune controls, then train and compare regression models (Linear, RF, KNN).\n
Scene 6: Manager debrief - explain which model best predicts show performance and why.\n
    """
)

with st.expander("Prompt 1: Return the first 5 rows", expanded=True):
    st.info("Story beat: Ben drops the raw tour log on your desk. First task: inspect the table before making any calls.")
    # st.dataframe(df.head(5), use_container_width=True)
    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Look at the first 5 rows of df"
            "\n#    Python idea: result = df.head(5)"
            "\n# 2. Save that preview in result"
            "\n# 3. Return result"
        ),
        runnable_code=(
            "# Step 1: Preview the first 5 rows"
            "\nresult = df.head(5)"
            "\n# Step 2: Print so you can see it in Output"
            "\nprint(result)"
        ),
        prompt_key="p1",
        challenge_question="Who is the Vice President & Chief Information Officer for Cat Financial IT?",
        challenge_answer="Chaille Becker".capitalize().strip(),
    )
    render_code_runner(
        label="Write your code for Prompt 1",
        key="p1",
        df=df,
        perf=perf,
    )

with st.expander("Prompt 2: Run simple statistics", expanded=False):
    st.info("Story beat: Ben wants a pre-rehearsal pulse check. Summarize score ranges, spread, and typical values.")
    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Create a summary table for df"
            "\n#    Python idea: summary_table = df.describe(include=\"all\")"
            "\n# 2. Check values like count, mean, std, min, and max"
            "\n# 3. Save the summary table in result"
            "\n#    Python idea: result = summary_table"
            "\n# 4. Return result"
        ),
        runnable_code=(
            "# Step 1: Create summary statistics for the dataframe"
            "\nresult = df.describe(include=\"all\")"
            "\n# Step 2: Print the summary"
            "\nprint(result)"
        ),
        prompt_key="p2_stats",
        challenge_question="What is the name of the coffee shop?",
        challenge_answer="Proving Grounds".capitalize().strip(),
    )
    render_code_runner(
        label="Write your code for Prompt 2",
        key="p2_stats",
        df=df,
        perf=perf,
    )

    st.markdown("### Additional challenge: Remove outliers from SCORE")
    st.caption("After checking summary stats, clean extreme SCORE values using an IQR-style rule.")
    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Find Q1 for SCORE"
            "\n#    Python idea: q1 = df['SCORE'].quantile(0.25)"
            "\n# 2. Find Q3 for SCORE"
            "\n#    Python idea: q3 = df['SCORE'].quantile(0.75)"
            "\n# 3. Compute IQR"
            "\n#    Python idea: iqr = q3 - q1"
            "\n# 4. Build the lower and upper score bounds"
            "\n#    Python idea: lower = q1 - 1.5 * iqr"
            "\n#    Python idea: upper = q3 + 1.5 * iqr"
            "\n# 5. Keep only rows where SCORE stays inside those bounds"
            "\n#    Python shortcut: result = df[(df['SCORE'] >= lower) & (df['SCORE'] <= upper)]"
            "\n# 6. Return result"
        ),
        runnable_code=(
            "# Step 0: Track how many rows we started with"
            "\nrows_before = len(df)"
            "\n# Step 1: Find quartiles for SCORE"
            "\nq1 = df['SCORE'].quantile(0.25)"
            "\nq3 = df['SCORE'].quantile(0.75)"
            "\n# Step 2: Calculate IQR and outlier bounds"
            "\niqr = q3 - q1"
            "\nlower = q1 - 1.5 * iqr"
            "\nupper = q3 + 1.5 * iqr"
            "\n# Step 3: Keep only rows inside the bounds"
            "\nresult = df[(df['SCORE'] >= lower) & (df['SCORE'] <= upper)]"
            "\n# Step 4: Show how many rows were removed"
            "\nprint('Rows before:', rows_before)"
            "\nprint('Rows after:', len(result))"
        ),
        prompt_key="p2_outliers",
        challenge_question="How many ERGs do we have at Cat Financial?",
        challenge_answer="10".strip(),
    )
    render_code_runner(
        label="Write your code for Prompt 2 (Outlier removal)",
        key="p2_outliers",
        df=df,
        perf=perf,
    )

with st.expander("Prompt 3: Drop columns not needed for modeling", expanded=False):
    st.info("Story beat: Build a fair pre-show playbook. Remove columns that give away the final score or expose personal details.")
    suggested_drop = [
        c
        for c in ["PERFORMANCE_ID", "PLAYER_ID", "PERFORMANCE_PLAYER_ID", "PLAYER_NAME", "PHONE_NUMBER", "EMAIL", "RATING"]
        if c in df.columns
    ]
    st.write("Current columns:")
    st.write(list(df.columns))
    st.write("Columns to consider dropping:")
    st.write(suggested_drop)

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Make a list of columns you want to remove"
            "\n#    Python idea: drop_list = [ ... ]"
            "\n#    Use IDs, personal info, and columns that give away the final score"
            "\n# 2. Keep only the column names that actually exist in df"
            "\n#    Python idea: safe_drop_list = [c for c in drop_list if c in df.columns]"
            "\n# 3. Drop those columns from df"
            "\n#    Python idea: result = df.drop(columns=safe_drop_list)"
            "\n# 4. Return result"
        ),
        runnable_code=(
            "# Step 1: Pick columns to remove"
            "\ndrop_list = [\"PERFORMANCE_ID\", \"PLAYER_ID\", \"PERFORMANCE_PLAYER_ID\", \"PLAYER_NAME\", \"PHONE_NUMBER\", \"EMAIL\", \"RATING\"]"
            "\n# Step 2: Keep only column names that exist in this dataset"
            "\nsafe_drop_list = [c for c in drop_list if c in df.columns]"
            "\n# Step 3: Drop those columns"
            "\nresult = df.drop(columns=safe_drop_list)"
            "\n# Step 4: Preview the cleaned dataframe"
            "\nprint(result.head())"
        ),
        prompt_key="p3_drop",
        challenge_question="What year was Caterpillar Inc. founded?",
        challenge_answer="1925",
    )
    render_code_runner(
        label="Write your code for Prompt 3",
        key="p3_drop",
        df=df,
        perf=perf,
    )

with st.expander("Prompt 4: Define target and candidate features", expanded=False):
    st.info("Story beat: Assume Prompt 3 already cleaned the table. Your job now is to name the prediction target (`SCORE`) and list the remaining columns you can use as features.")
    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Assume Prompt 3 already removed the bad columns"
            "\n# 2. Set the target column to SCORE"
            "\n#    Python idea: target_column = \"SCORE\""
            "\n# 3. Treat every other remaining column as a candidate feature"
            "\n#    Python idea: feature_columns = [c for c in df.columns if c != target_column]"
            "\n# 4. Save a quick summary in result"
            "\n#    Python idea: result = {\"target_column\": target_column, \"feature_columns\": feature_columns}"
            "\n# 5. Return result"
        ),
        runnable_code=(
            "# Step 1: Assume df is already cleaned from Prompt 3"
            "\nclean_df = df.copy()"
            "\n# Step 2: Choose the target column we want to predict"
            "\ntarget_column = \"SCORE\""
            "\n# Step 3: Use every other remaining column as a feature"
            "\nfeature_columns = [c for c in clean_df.columns if c != target_column]"
            "\n# Step 4: Build an easy-to-read summary"
            "\nresult = pd.DataFrame({"
            "\n    \"target_column\": [target_column],"
            "\n    \"feature_count\": [len(feature_columns)],"
            "\n    \"example_feature_1\": [feature_columns[0] if len(feature_columns) > 0 else \"N/A\"],"
            "\n    \"example_feature_2\": [feature_columns[1] if len(feature_columns) > 1 else \"N/A\"],"
            "\n    \"example_feature_3\": [feature_columns[2] if len(feature_columns) > 2 else \"N/A\"]"
            "\n})"
            "\nprint(result)"
        ),
        prompt_key="p4_shape",
        challenge_question="Who is the President and Chief Executive Officer of Caterpillar Financial Services Corporation",
        challenge_answer="Dave Walton".capitalize().strip(),
    )
    render_code_runner(
        label="Write your code for Prompt 4",
        key="p4_shape",
        df=df,
        perf=perf,
    )

with st.expander("Prompt 5: Simple analytics and modeling", expanded=False):
    st.info("Story beat: This is your rehearsal lab. Set modeling controls, run benchmark tryouts, and choose Ben's best pre-show predictor.")
    with st.expander("Quick bonus trivia (just for fun)", expanded=False):
        st.caption("Not related to data or scoring. This is just a fun break question.")
        st.markdown("How many floors do we have total?")
        bonus_answer = st.text_input("Your answer", key="bonus_floor_answer")

        normalized_bonus = bonus_answer.strip().lower()
        if normalized_bonus:
            if normalized_bonus in {"17", "18"}:
                st.success("Nice. Correct answer.")
            else:
                st.warning("Not this time. Accepted answers are 17 or 18.")

    st.markdown("### Rehearsal controls (set these before training)")
    st.caption("These controls are part of the modeling prompt and affect how the benchmark is run.")
    remove_outliers = st.toggle(
        "Remove unusual score outliers",
        value=remove_outliers,
        key="model_remove_outliers",
    )
    test_size_pct = st.slider(
        "Test split (%)",
        min_value=10,
        max_value=40,
        value=test_size_pct,
        step=5,
        key="model_test_size_pct",
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=9999,
        value=random_seed,
        step=1,
        key="model_random_seed",
    )

    try:
        perf_for_model = build_performance_table(df, remove_outliers=remove_outliers)
    except Exception as exc:
        st.error(f"Unable to build performance-level table for modeling: {exc}")
        perf_for_model = None

    st.markdown("### What to do before running models")
    st.markdown("- Confirm you are modeling at one row per performance (`perf_for_model`), not raw player-level rows.")
    st.markdown("- Keep the same train/test split for all models so comparison is fair.")
    st.markdown("- Include the NaiveMean baseline to show whether ML adds value.")

    st.markdown("### How to read the metrics")
    st.markdown("- MAE: average absolute error in score units (lower is better).")
    st.markdown("- RMSE: stronger penalty for big misses (lower is better).")
    st.markdown("- R2: proportion of variance explained (higher is better).")
    st.markdown("- Interview tip: choose your winner primarily by **Test_RMSE**, then support with MAE and R2.")

    st.markdown("### What to report after running")
    st.markdown("1. Which model had the lowest Test_RMSE.")
    st.markdown("2. How much better it was than NaiveMean.")
    st.markdown("3. Whether train and test scores suggest underfitting or overfitting.")

    st.markdown("Fine-tuning hints")
    st.markdown("- Random Forest: tune `n_estimators`, `max_depth`, and `min_samples_leaf`.")
    st.markdown("- KNN: tune `n_neighbors` and `weights`.")
    st.markdown("- Choose best model primarily by lowest Test_RMSE.")
    st.markdown("- If overfitting: reduce model complexity or add more data.")
    st.markdown("- If underfitting: increase model flexibility or add better features.")

    run_models = st.button("Run benchmark models", type="primary", disabled=perf_for_model is None)
    if run_models and perf_for_model is not None:
        metrics_df, trained_models, split_data = evaluate_holdout(
            perf=perf_for_model,
            model_names=["LinearRegression", "RandomForestRegressor", "KNeighborsRegressor"],
            test_size=test_size_pct / 100.0,
            seed=int(random_seed),
        )
        st.session_state.holdout_metrics = metrics_df
        st.session_state.holdout_models = trained_models
        st.session_state.split_data = split_data

    if st.session_state.holdout_metrics is not None:
        metrics_df = st.session_state.holdout_metrics
        st.dataframe(metrics_df, use_container_width=True)

        chart_df = metrics_df[metrics_df["Model"] != "NaiveMean"].copy()
        if not chart_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.barplot(data=chart_df, x="Model", y="Test_RMSE", ax=axes[0], hue="Model", legend=False)
            axes[0].set_title("Test RMSE")
            axes[0].tick_params(axis="x", rotation=20)

            sns.barplot(data=chart_df, x="Model", y="Test_R2", ax=axes[1], hue="Model", legend=False)
            axes[1].set_title("Test R2")
            axes[1].tick_params(axis="x", rotation=20)
            st.pyplot(fig)

with st.expander("Prompt 6 (Challenging): Analyze the best model and why", expanded=False):
    st.info("Story beat: Ben's manager needs a decision memo before tonight's show. Explain why the winning model works and where it still misses.")
    st.markdown("### Analysis framework (use this structure)")
    st.markdown("1. **Winner and evidence**: name the best model and quote Test_RMSE, MAE, and R2.")
    st.markdown("2. **Why it likely won**: explain the pattern it can capture (linear trend, local neighbors, nonlinear interactions).")
    st.markdown("3. **What drove predictions**: use permutation importance to name top features.")
    st.markdown("4. **Where it fails**: use largest residuals to describe repeated error patterns.")
    st.markdown("5. **Next improvement**: give one concrete tuning or feature-engineering step.")

    with st.expander("Example strong analysis answer", expanded=False):
        st.markdown(
            "- Random Forest achieved the lowest Test_RMSE, so it is the best holdout model on this split.\n"
            "- It likely won because score patterns are nonlinear and involve feature interactions.\n"
            "- Top importance features suggest team composition and song characteristics drive prediction most.\n"
            "- Largest residuals cluster on rare high-score performances, so the model struggles on edge cases.\n"
            "- Next step: tune tree depth/min leaf and test adding richer event-level features."
        )

    metrics_df = st.session_state.holdout_metrics
    split_data = st.session_state.split_data

    if metrics_df is None or split_data is None:
        st.warning("Run Prompt 5 first.")
    else:
        non_naive = metrics_df[metrics_df["Model"] != "NaiveMean"]
        if non_naive.empty:
            st.warning("No trained non-baseline model found.")
        else:
            best_row = non_naive.sort_values("Test_RMSE", ascending=True).iloc[0]
            best_name = str(best_row["Model"])
            st.success(f"Best model: {best_name} (Test_RMSE={float(best_row['Test_RMSE']):.2f})")

            best_model = st.session_state.holdout_models.get(best_name)
            if best_model is not None:
                x_test = split_data["x_test"]
                y_test = split_data["y_test"]

                importances = permutation_importance(
                    best_model,
                    x_test,
                    y_test,
                    n_repeats=12,
                    random_state=int(random_seed),
                    scoring="neg_root_mean_squared_error",
                )
                imp_df = pd.DataFrame(
                    {"Feature": x_test.columns, "Importance": importances.importances_mean}
                ).sort_values("Importance", ascending=False)
                st.markdown("Top feature importances")
                st.dataframe(imp_df.head(10), use_container_width=True)

                y_pred = best_model.predict(x_test)
                residual_df = pd.DataFrame(
                    {
                        "Actual": y_test.values,
                        "Predicted": y_pred,
                        "Residual": y_test.values - y_pred,
                    },
                    index=y_test.index,
                )
                residual_df["Abs_Error"] = residual_df["Residual"].abs()
                st.markdown("Largest residual errors")
                st.dataframe(residual_df.sort_values("Abs_Error", ascending=False).head(10), use_container_width=True)

            render_prompt_help(
                example_code=(
                    "# Pseudo-code recipe"
                    "\n# 1. Find the model with the lowest Test_RMSE"
                    "\n#    Python idea: best_row = metrics_df.sort_values('Test_RMSE').iloc[0]"
                    "\n#    Use test metrics, not training metrics, for the final choice"
                    "\n# 2. Collect the key test metrics for that model"
                    "\n#    Python idea: key_metrics = { ... }"
                    "\n# 3. Grab the top important features"
                    "\n#    Python idea: top_features = imp_df.head(3)"
                    "\n# 4. Look at the biggest residual errors to describe the error pattern"
                    "\n#    Python idea: error_pattern = residual_df.sort_values(...).head(...)"
                    "\n# 5. Add one next-step improvement idea"
                    "\n# 6. Save the summary in result and return it"
                    "\n#    Python idea: result = { ... }"
                ),
                runnable_code=(
                    "# Step 1: Select the best non-baseline model by lowest Test_RMSE"
                    "\nbest_row = metrics_df[metrics_df['Model'] != 'NaiveMean'].sort_values('Test_RMSE').iloc[0]"
                    "\n# Step 2: Capture top feature drivers"
                    "\ntop_drivers = imp_df.head(3)['Feature'].tolist()"
                    "\n# Step 3: Find the largest prediction miss"
                    "\nworst_error = residual_df.sort_values('Abs_Error', ascending=False).iloc[0]"
                    "\n# Step 4: Build a clear summary object"
                    "\nresult = {"
                    "\n    'best_model': best_row['Model'],"
                    "\n    'test_rmse': float(best_row['Test_RMSE']),"
                    "\n    'top_drivers': top_drivers,"
                    "\n    'largest_error_actual': float(worst_error['Actual']),"
                    "\n    'largest_error_predicted': float(worst_error['Predicted'])"
                    "\n}"
                    "\nprint(result)"
                ),
                prompt_key="p6_analysis",
                challenge_question="Name at least one Enterprise Architect (EA) and one Domain Architect (DA). \nFind the champion to confirm the name to get the hint.",
                challenge_answer="Notch".capitalize().strip(),
            )
            st.markdown("### Final answer checklist")
            st.markdown("- Mention the winning model and at least two metrics.")
            st.markdown("- Mention at least two important features.")
            st.markdown("- Mention one recurring error pattern from residuals.")
            st.markdown("- Mention one concrete next step.")
            st.text_area(
                "Your analysis answer",
                key="analysis_answer",
                height=120,
            )

st.divider()
render_team_results_store(
    remove_outliers=remove_outliers,
    test_size_pct=int(test_size_pct),
    random_seed=int(random_seed),
)

st.caption("Intern Game Day 2026 | Ben's Rock Band")
