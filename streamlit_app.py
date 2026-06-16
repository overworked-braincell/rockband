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


CHALLENGE_DEFAULTS = {
    "p1": {
        "question": "Who is the Vice President & Chief Information Officer for Cat Financial IT?",
        "answer": "Chaille Becker",
    },
    "p2_stats": {
        "question": "What is the name of the coffee shop?",
        "answer": "Proving Grounds",
    },
    "p2_outliers": {
        "question": "How many ERGs do we have at Cat Financial?",
        "answer": "10",
    },
    "p3_drop": {
        "question": "What year was Caterpillar Inc. founded?",
        "answer": "1925",
    },
    "p4_shape": {
        "question": "Who is the President and Chief Executive Officer of Caterpillar Financial Services Corporation",
        "answer": "Dave Walton",
    },
    "p5_prepare": {
        "question": "What is the convience store downstaris called?",
        "answer": "Stop Gap",
    },
    "p6_modeling": {
        "question": "Who is the Cat Financial EA this challenge is based off of?",
        "answer": "Ben Hocker",
    },
}

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
        "metrics_df": st.session_state.get("holdout_metrics"),
        "split_data": st.session_state.get("split_data"),
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
    challenge_question: str | None = None,
    challenge_answer: str | None = None,
) -> None:
    answer_key = f"unlock_answer_{prompt_key}"
    current_answer = str(st.session_state.get(answer_key, "")).strip()
    if prompt_key == "p1":
        penalty_points = 5
    elif prompt_key == "p4_shape":
        penalty_points = 1
    else:
        penalty_points = 3

    with st.expander("Pseudo-code", expanded=False):
        st.markdown("### Pseudo-code guidance")
        st.markdown("Use this as a simple recipe. Read one line, write one matching Python line.")
        st.code(example_code.strip(), language="python")

    has_challenge = bool(challenge_question) and bool(challenge_answer)

    if has_challenge:
        with st.expander("Runnable example (locked)", expanded=bool(current_answer)):
            st.markdown("### Challenge unlock")
            st.markdown("Answer the challenge question correctly to reveal the runnable reference code.")
            st.caption(f"Scoring rule: unlocking this runnable example applies a one-time -{penalty_points} point penalty.")
            st.markdown(challenge_question)
            user_answer = st.text_input(
                "Enter your answer to unlock",
                key=answer_key,
            )

            if user_answer.strip().lower() == challenge_answer.strip().lower():
                penalty_key = f"runnable_penalty_applied_{prompt_key}"
                if not st.session_state.get(penalty_key, False):
                    st.session_state.challenge_points = int(st.session_state.get("challenge_points", 0)) - penalty_points
                    st.session_state[penalty_key] = True
                    st.warning(f"Runnable example unlocked. -{penalty_points} points applied.")
                st.success("Correct. Runnable example unlocked.")
                st.markdown("Use this as a starter template and edit it in your own style.")
                st.code(runnable_code.strip(), language="python")
            elif user_answer.strip():
                st.warning("Not quite. Try again.")
    else:
        with st.expander("Runnable example", expanded=False):
            st.markdown("### Runnable reference code")
            st.markdown("Use this as a starter template and edit it in your own style.")
            st.code(runnable_code.strip(), language="python")


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


def compute_submission_score(
    metrics_df: pd.DataFrame | None,
    challenge_points: int,
) -> dict[str, float | str]:
    if metrics_df is None or metrics_df.empty:
        return {
            "score_total": 0.0,
            "score_improvement": 0.0,
            "score_generalization": 0.0,
            "score_model_quality": 0.0,
            "score_challenge_modifier": float(max(-10, min(10, challenge_points))),
            "score_grade": "N/A",
        }

    non_naive = metrics_df[metrics_df["Model"] != "NaiveMean"]
    naive = metrics_df[metrics_df["Model"] == "NaiveMean"]

    if non_naive.empty:
        return {
            "score_total": 0.0,
            "score_improvement": 0.0,
            "score_generalization": 0.0,
            "score_model_quality": 0.0,
            "score_challenge_modifier": float(max(-10, min(10, challenge_points))),
            "score_grade": "N/A",
        }

    best_row = non_naive.sort_values("Test_RMSE", ascending=True).iloc[0]

    naive_test_rmse = float(naive.iloc[0]["Test_RMSE"]) if not naive.empty else float(best_row["Test_RMSE"])
    best_test_rmse = float(best_row["Test_RMSE"])
    best_train_rmse = float(best_row["Train_RMSE"])
    best_test_r2 = float(best_row["Test_R2"])

    improvement_ratio = 0.0
    if naive_test_rmse > 0:
        improvement_ratio = max(0.0, (naive_test_rmse - best_test_rmse) / naive_test_rmse)

    improvement_points = min(55.0, (improvement_ratio / 0.35) * 55.0)
    model_quality_points = min(20.0, max(0.0, ((best_test_r2 + 1.0) / 2.0) * 20.0))

    gap_ratio = abs(best_train_rmse - best_test_rmse) / max(best_test_rmse, 1e-9)
    generalization_points = max(0.0, 20.0 - (40.0 * gap_ratio))

    challenge_modifier = float(max(-10, min(10, challenge_points)))

    total = max(
        0.0,
        min(
            100.0,
            improvement_points + model_quality_points + generalization_points + challenge_modifier,
        ),
    )

    if total >= 90:
        grade = "A"
    elif total >= 80:
        grade = "B"
    elif total >= 70:
        grade = "C"
    elif total >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "score_total": round(total, 1),
        "score_improvement": round(improvement_points, 1),
        "score_generalization": round(generalization_points, 1),
        "score_model_quality": round(model_quality_points, 1),
        "score_challenge_modifier": round(challenge_modifier, 1),
        "score_grade": grade,
    }


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

    challenge_points = int(st.session_state.get("challenge_points", 0))
    score_breakdown = compute_submission_score(metrics_df=metrics_df, challenge_points=challenge_points)

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
        "challenge_points": challenge_points,
        "score_total": score_breakdown["score_total"],
        "score_grade": score_breakdown["score_grade"],
        "score_improvement": score_breakdown["score_improvement"],
        "score_generalization": score_breakdown["score_generalization"],
        "score_model_quality": score_breakdown["score_model_quality"],
        "score_challenge_modifier": score_breakdown["score_challenge_modifier"],
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
    st.info(
        f"Submission score: {score_breakdown['score_total']}/100 (Grade {score_breakdown['score_grade']})"
    )
    st.dataframe(new_row_df, use_container_width=True)


if "holdout_metrics" not in st.session_state:
    st.session_state.holdout_metrics = None
if "holdout_models" not in st.session_state:
    st.session_state.holdout_models = {}
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "challenge_points" not in st.session_state:
    st.session_state.challenge_points = 0
for prompt_id, challenge_cfg in CHALLENGE_DEFAULTS.items():
    q_key = f"challenge_question_{prompt_id}"
    a_key = f"challenge_answer_{prompt_id}"
    if q_key not in st.session_state:
        st.session_state[q_key] = challenge_cfg["question"]
    if a_key not in st.session_state:
        st.session_state[a_key] = challenge_cfg["answer"]


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
    st.markdown("### How to Read the Metrics")
    st.markdown("- MAE: average absolute error in score units (lower is better).")
    st.markdown("- RMSE: stronger penalty for big misses (lower is better).")
    st.markdown("- R2: proportion of variance explained (higher is better).")
    st.markdown("- Pick winners mainly by **Test_RMSE**, then support with MAE and R2.")

    st.markdown("### Challenge Settings")
    st.caption("Edit challenge question and answer per prompt.")
    with st.expander("Configure challenge prompts", expanded=False):
        st.markdown("Prompt 1")
        st.session_state["challenge_question_p1"] = st.text_input(
            "Prompt 1 challenge question",
            value=st.session_state["challenge_question_p1"],
            key="challenge_question_p1_input",
        )
        st.session_state["challenge_answer_p1"] = st.text_input(
            "Prompt 1 challenge answer",
            value=st.session_state["challenge_answer_p1"],
            key="challenge_answer_p1_input",
        )

        st.markdown("Prompt 2")
        st.session_state["challenge_question_p2_stats"] = st.text_input(
            "Prompt 2 challenge question",
            value=st.session_state["challenge_question_p2_stats"],
            key="challenge_question_p2_stats_input",
        )
        st.session_state["challenge_answer_p2_stats"] = st.text_input(
            "Prompt 2 challenge answer",
            value=st.session_state["challenge_answer_p2_stats"],
            key="challenge_answer_p2_stats_input",
        )

        st.markdown("Prompt 2 (Outlier removal)")
        st.session_state["challenge_question_p2_outliers"] = st.text_input(
            "Prompt 2 outlier challenge question",
            value=st.session_state["challenge_question_p2_outliers"],
            key="challenge_question_p2_outliers_input",
        )
        st.session_state["challenge_answer_p2_outliers"] = st.text_input(
            "Prompt 2 outlier challenge answer",
            value=st.session_state["challenge_answer_p2_outliers"],
            key="challenge_answer_p2_outliers_input",
        )

        st.markdown("Prompt 3")
        st.session_state["challenge_question_p3_drop"] = st.text_input(
            "Prompt 3 challenge question",
            value=st.session_state["challenge_question_p3_drop"],
            key="challenge_question_p3_drop_input",
        )
        st.session_state["challenge_answer_p3_drop"] = st.text_input(
            "Prompt 3 challenge answer",
            value=st.session_state["challenge_answer_p3_drop"],
            key="challenge_answer_p3_drop_input",
        )

        st.markdown("Prompt 4")
        st.session_state["challenge_question_p4_shape"] = st.text_input(
            "Prompt 4 challenge question",
            value=st.session_state["challenge_question_p4_shape"],
            key="challenge_question_p4_shape_input",
        )
        st.session_state["challenge_answer_p4_shape"] = st.text_input(
            "Prompt 4 challenge answer",
            value=st.session_state["challenge_answer_p4_shape"],
            key="challenge_answer_p4_shape_input",
        )

        st.markdown("Prompt 5")
        st.session_state["challenge_question_p5_prepare"] = st.text_input(
            "Prompt 5 challenge question",
            value=st.session_state["challenge_question_p5_prepare"],
            key="challenge_question_p5_prepare_input",
        )
        st.session_state["challenge_answer_p5_prepare"] = st.text_input(
            "Prompt 5 challenge answer",
            value=st.session_state["challenge_answer_p5_prepare"],
            key="challenge_answer_p5_prepare_input",
        )

        st.markdown("Prompt 6")
        st.session_state["challenge_question_p6_modeling"] = st.text_input(
            "Prompt 6 challenge question",
            value=st.session_state["challenge_question_p6_modeling"],
            key="challenge_question_p6_modeling_input",
        )
        st.session_state["challenge_answer_p6_modeling"] = st.text_input(
            "Prompt 6 challenge answer",
            value=st.session_state["challenge_answer_p6_modeling"],
            key="challenge_answer_p6_modeling_input",
        )

p1_challenge_question = str(st.session_state["challenge_question_p1"]).strip() or CHALLENGE_DEFAULTS["p1"]["question"]
p1_challenge_answer = str(st.session_state["challenge_answer_p1"]).strip() or CHALLENGE_DEFAULTS["p1"]["answer"]
p2_challenge_question = str(st.session_state["challenge_question_p2_stats"]).strip() or CHALLENGE_DEFAULTS["p2_stats"]["question"]
p2_challenge_answer = str(st.session_state["challenge_answer_p2_stats"]).strip() or CHALLENGE_DEFAULTS["p2_stats"]["answer"]
p2_outlier_challenge_question = str(st.session_state["challenge_question_p2_outliers"]).strip() or CHALLENGE_DEFAULTS["p2_outliers"]["question"]
p2_outlier_challenge_answer = str(st.session_state["challenge_answer_p2_outliers"]).strip() or CHALLENGE_DEFAULTS["p2_outliers"]["answer"]
p3_challenge_question = str(st.session_state["challenge_question_p3_drop"]).strip() or CHALLENGE_DEFAULTS["p3_drop"]["question"]
p3_challenge_answer = str(st.session_state["challenge_answer_p3_drop"]).strip() or CHALLENGE_DEFAULTS["p3_drop"]["answer"]
p4_challenge_question = str(st.session_state["challenge_question_p4_shape"]).strip() or CHALLENGE_DEFAULTS["p4_shape"]["question"]
p4_challenge_answer = str(st.session_state["challenge_answer_p4_shape"]).strip() or CHALLENGE_DEFAULTS["p4_shape"]["answer"]
p5_challenge_question = str(st.session_state["challenge_question_p5_prepare"]).strip() or CHALLENGE_DEFAULTS["p5_prepare"]["question"]
p5_challenge_answer = str(st.session_state["challenge_answer_p5_prepare"]).strip() or CHALLENGE_DEFAULTS["p5_prepare"]["answer"]
p6_challenge_question = str(st.session_state["challenge_question_p6_modeling"]).strip() or CHALLENGE_DEFAULTS["p6_modeling"]["question"]
p6_challenge_answer = str(st.session_state["challenge_answer_p6_modeling"]).strip() or CHALLENGE_DEFAULTS["p6_modeling"]["answer"]

st.markdown(
    """
Scene 1: Backstage intake - open Ben's gig history and preview rows.\n
Scene 2: Feature planning - define target and choose candidate predictors.\n
Scene 3: Crowd pulse check - profile statistics to understand score behavior.\n
Scene 4: Setlist cleanup - keep only useful columns for fair prediction.\n
Scene 5: Data prep lab - build model-ready inputs and split train/test fairly.\n
Scene 6: Rehearsal benchmark - train and compare regression models (Linear, RF, KNN).\n
Scene 7: Manager debrief - explain which model best predicts show performance and why.\n
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
        challenge_question=p1_challenge_question,
        challenge_answer=p1_challenge_answer,
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
        challenge_question=p2_challenge_question,
        challenge_answer=p2_challenge_answer,
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
        challenge_question=p2_outlier_challenge_question,
        challenge_answer=p2_outlier_challenge_answer,
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
        challenge_question=p3_challenge_question,
        challenge_answer=p3_challenge_answer,
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
        challenge_question=p4_challenge_question,
        challenge_answer=p4_challenge_answer,
    )
    render_code_runner(
        label="Write your code for Prompt 4",
        key="p4_shape",
        df=df,
        perf=perf,
    )

with st.expander("Prompt 5: Prepare model-ready data", expanded=False):
    st.info("Story beat: before the rehearsal lab starts, prepare clean modeling inputs and a reliable train/test split.")

    st.markdown("### Modeling setup")
    st.caption("These controls affect Prompt 6 benchmarking as well.")
    remove_outliers = st.toggle(
        "Remove score outliers",
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
        st.error(f"Unable to build performance-level table: {exc}")
        perf_for_model = None

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe\n"
            "# 1. Start from perf_for_model\n"
            "# 2. Remove leakage columns\n"
            "# 3. Set y to SCORE\n"
            "# 4. Set X to all remaining columns except SCORE\n"
            "# 5. Split into train/test\n"
            "# 6. Save quick shape checks in result\n"
        ),
        runnable_code=(
            "drop_cols = ['PERFORMANCE_ID', 'RATING', 'PERFORMANCE_DATETIME']\n"
            "safe_cols = [c for c in drop_cols if c in perf_for_model.columns]\n"
            "X = perf_for_model.drop(columns=safe_cols + ['SCORE']).fillna(0)\n"
            "y = perf_for_model['SCORE'].fillna(0)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "result = {'x_train': X_train.shape, 'x_test': X_test.shape, 'y_train': y_train.shape, 'y_test': y_test.shape}\n"
            "print(result)\n"
        ),
        prompt_key="p5_prepare",
        challenge_question=p5_challenge_question,
        challenge_answer=p5_challenge_answer,
    )
    render_code_runner(
        label="Write your code for Prompt 5",
        key="p5_prepare",
        df=df,
        perf=perf_for_model if perf_for_model is not None else df,
    )


with st.expander("Prompt 6: Train and compare regression models", expanded=False):
    st.info("Story beat: run the rehearsal benchmark. Train baseline + three regressors and compare fairly.")

    p6_runnable_code = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from sklearn.neighbors import KNeighborsRegressor\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        "import numpy as np\n"
        "\n"
        "drop_cols = ['PERFORMANCE_ID', 'RATING', 'PERFORMANCE_DATETIME']\n"
        "safe_cols = [c for c in drop_cols if c in perf_for_model.columns]\n"
        "X = perf_for_model.drop(columns=safe_cols + ['SCORE']).fillna(0)\n"
        "y = perf_for_model['SCORE'].fillna(0)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "\n"
        "rows = []\n"
        "naive_pred_train = np.full(len(y_train), y_train.mean())\n"
        "naive_pred_test = np.full(len(y_test), y_train.mean())\n"
        "rows.append({\n"
        "    'Model': 'NaiveMean',\n"
        "    'Train_MAE': round(mean_absolute_error(y_train, naive_pred_train), 2),\n"
        "    'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, naive_pred_train)), 2),\n"
        "    'Train_R2': round(r2_score(y_train, naive_pred_train), 4),\n"
        "    'Test_MAE': round(mean_absolute_error(y_test, naive_pred_test), 2),\n"
        "    'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, naive_pred_test)), 2),\n"
        "    'Test_R2': round(r2_score(y_test, naive_pred_test), 4),\n"
        "})\n"
        "\n"
        "models = [\n"
        "    ('LinearRegression', LinearRegression()),\n"
        "    ('RandomForestRegressor', RandomForestRegressor(n_estimators=100, random_state=42)),\n"
        "    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=5))\n"
        "]\n"
        "\n"
        "for name, model in models:\n"
        "    model.fit(X_train, y_train)\n"
        "    pred_train = model.predict(X_train)\n"
        "    pred_test = model.predict(X_test)\n"
        "    rows.append({\n"
        "        'Model': name,\n"
        "        'Train_MAE': round(mean_absolute_error(y_train, pred_train), 2),\n"
        "        'Train_RMSE': round(np.sqrt(mean_squared_error(y_train, pred_train)), 2),\n"
        "        'Train_R2': round(r2_score(y_train, pred_train), 4),\n"
        "        'Test_MAE': round(mean_absolute_error(y_test, pred_test), 2),\n"
        "        'Test_RMSE': round(np.sqrt(mean_squared_error(y_test, pred_test)), 2),\n"
        "        'Test_R2': round(r2_score(y_test, pred_test), 4),\n"
        "    })\n"
        "\n"
        "result = pd.DataFrame(rows).sort_values('Test_RMSE', ascending=True).reset_index(drop=True)\n"
        "print(result)\n"
    )

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe\n"
            "# 1. Rebuild X, y and split from perf_for_model\n"
            "# 2. Add NaiveMean baseline metrics\n"
            "# 3. Train LinearRegression, RandomForest, KNN\n"
            "# 4. Compute Train/Test MAE, RMSE, R2 for each\n"
            "# 5. Sort by Test_RMSE and save as result\n"
        ),
        runnable_code=p6_runnable_code,
        prompt_key="p6_modeling",
        challenge_question=p6_challenge_question,
        challenge_answer=p6_challenge_answer,
    )

    if st.button("Insert starter template into Prompt 6 editor", key="insert_p6_template_code"):
        st.session_state["prompt_code_p6_modeling"] = p6_runnable_code
        st.success("Starter template inserted into Prompt 6 editor.")

    render_code_runner(
        label="Write your code for Prompt 6",
        key="p6_modeling",
        df=df,
        perf=perf_for_model if 'perf_for_model' in locals() and perf_for_model is not None else df,
    )

    if st.button("Use built-in benchmark and send to Prompt 7", key="p6_builtin_benchmark", disabled=('perf_for_model' not in locals() or perf_for_model is None)):
        if 'perf_for_model' in locals() and perf_for_model is not None:
            metrics_df, trained_models, split_data = evaluate_holdout(
                perf=perf_for_model,
                model_names=["LinearRegression", "RandomForestRegressor", "KNeighborsRegressor"],
                test_size=test_size_pct / 100.0,
                seed=int(random_seed),
            )
            st.session_state.holdout_metrics = metrics_df
            st.session_state.holdout_models = trained_models
            st.session_state.split_data = split_data
            st.success("Benchmark saved. Prompt 7 is now ready.")
            st.dataframe(metrics_df, use_container_width=True)


with st.expander("Prompt 7 (Challenging): Interpret your best model", expanded=False):
    st.info("Story beat: Ben's manager needs a decision memo before tonight's show. Explain your model choice, why it works, and where it still struggles.")

    metrics_df = st.session_state.holdout_metrics
    if metrics_df is not None and not metrics_df.empty:
        non_naive = metrics_df[metrics_df["Model"] != "NaiveMean"]
        if not non_naive.empty:
            best_row = non_naive.sort_values("Test_RMSE", ascending=True).iloc[0]
            st.success(
                f"Current best model from saved metrics: {best_row['Model']} | Test_RMSE={float(best_row['Test_RMSE']):.2f}"
            )
    
    st.markdown("### Your task")
    st.markdown("Write a **3-5 sentence executive summary** that covers:")
    st.markdown("1. Which model won and by how much (cite Test_RMSE vs. NaiveMean baseline).")
    st.markdown("2. Why it likely outperforms alternatives (e.g., captures nonlinearity, local patterns).")
    st.markdown("3. Evidence of overfitting or underfitting (compare Train vs. Test metrics).")
    st.markdown("4. One weakness or failure mode (which types of performances does it struggle with?).")
    st.markdown("5. One concrete next step (tuning idea, new feature, or data collection).")
    
    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe\n"
            "# 1. Extract the winning model row from your Prompt 6 results\n"
            "#    Python idea: best = result.sort_values('Test_RMSE').iloc[0]\n"
            "# 2. Compute improvement over baseline\n"
            "#    Python idea: baseline_rmse = result[result['Model'] == 'NaiveMean']['Test_RMSE']\n"
            "# 3. Compare Train_RMSE vs. Test_RMSE to diagnose fit\n"
            "# 4. Consider which prediction types might fail\n"
            "# 5. Draft your written summary\n"
            "# 6. Save as result (string)\n"
        ),
        runnable_code=(
            "# Step 1: Summarize your Prompt 6 results\n"
            "best_row = result.sort_values('Test_RMSE').iloc[0]\n"
            "baseline_row = result[result['Model'] == 'NaiveMean'].iloc[0]\n"
            "\n"
            "# Step 2: Calculate improvement\n"
            "improvement_pct = ((baseline_row['Test_RMSE'] - best_row['Test_RMSE']) / baseline_row['Test_RMSE'] * 100)\n"
            "\n"
            "# Step 3: Assess overfitting\n"
            "train_rmse = best_row['Train_RMSE']\n"
            "test_rmse = best_row['Test_RMSE']\n"
            "is_overfitting = train_rmse < test_rmse * 0.8  # Train much better than test\n"
            "\n"
            "# Step 4: Build your summary\n"
            "result = f\"\"\"\n"
            "Best Model: {best_row['Model']}\n"
            "Test RMSE: {best_row['Test_RMSE']:.2f} (vs. {baseline_row['Test_RMSE']:.2f} baseline, {improvement_pct:.1f}% better)\n"
            "Train RMSE: {train_rmse:.2f} vs. Test RMSE: {test_rmse:.2f} -> Overfitting: {is_overfitting}\n"
            "R2 Score: Train={best_row['Train_R2']:.3f}, Test={best_row['Test_R2']:.3f}\n\"\"\"\n"
            "print(result)\n"
        ),
        prompt_key="p7_analysis",
    )

    st.markdown("### Code prompt (optional, but recommended)")
    st.caption("Use this to compute your summary from `metrics_df` before writing your final interpretation.")
    render_code_runner(
        label="Write code for Prompt 7 analysis (use metrics_df, then set result)",
        key="p7_analysis_code",
        df=df,
        perf=perf,
    )

    if st.button("Insert writing template into Prompt 7 answer", key="insert_p7_template"):
        st.session_state["analysis_answer"] = (
            "1) Winning model and evidence: <model_name> had Test_RMSE=<value>, improving over NaiveMean by <value>%.\n"
            "2) Why it won: <short reason tied to pattern type>.\n"
            "3) Fit diagnosis: Train vs Test gap suggests <overfitting/underfitting>.\n"
            "4) Failure mode: biggest misses happened on <type of performances>.\n"
            "5) Next step: I will <one concrete tuning or feature step>."
        )
        st.success("Prompt 7 writing template inserted.")

    st.caption("Sentence starters: The winning model was... | Compared with baseline... | A risk I noticed is... | Next I would...")
    
    st.markdown("### Your interpretation")
    st.text_area(
        "Write your 3-5 sentence model interpretation and next-step recommendation",
        key="analysis_answer",
        height=120,
        placeholder="Example: RandomForest achieved Test RMSE of 8.2 vs. baseline 12.0 (32% improvement). "
                    "Linear Regression scored 10.1, suggesting score patterns are nonlinear. "
                    "Train RMSE was 5.8 vs. Test 8.2, indicating modest overfitting. "
                    "Residuals suggest the model struggles on rare high-score performances. "
                    "Next: tune max_depth and retrain with cross-validation for stability.",
    )
    
    st.markdown("### Self-check")
    st.markdown("✓ Did you cite specific test metrics from Prompt 6?")
    st.markdown("✓ Did you compare to the NaiveMean baseline?")
    st.markdown("✓ Did you assess train/test gap (overfitting risk)?")
    st.markdown("✓ Did you identify at least one failure mode?")
    st.markdown("✓ Did you propose one concrete next step?")

st.divider()
render_team_results_store(
    remove_outliers=remove_outliers,
    test_size_pct=int(test_size_pct),
    random_seed=int(random_seed),
)

st.caption("Intern Game Day 2026 | Ben's Rock Band")
