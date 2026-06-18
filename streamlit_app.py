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
        "question": "What is the convenience store downstairs refer as?",
        "answer": "Stop Gap",
    },
    "p6_modeling": {
        "question": "Who is the Cat Financial EA this challenge is based off of?",
        "answer": "Ben Hocker",
    },
}

CHALLENGE_PENALTIES = {
    "p1": 5.0,
    "p2_stats": 3.0,
    "p2_outliers": 0.0,
    "p3_drop": 2.0,
    "p4_shape": 1.0,
    "p5_prepare": 0.0,
    "p6_modeling": 0.5,
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
    include_baseline: bool = False,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, pd.DataFrame | pd.Series]]:
    feature_cols, cat_cols, num_cols = get_feature_columns(perf)
    x = perf[feature_cols]
    y = perf["SCORE"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Pipeline] = {}

    if include_baseline:
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
        "train_test_split": train_test_split,
        "df": df.copy(),
        "perf": perf.copy(),
        "perf_for_model": perf.copy(),
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


_STARTER_TEMPLATE_HINT = "Use this as a starter template and edit it in your own style."


def _render_free_runnable(
    runnable_code: str,
    prompt_key: str,
    penalty_points: float = 0.0,
) -> None:
    with st.expander("Runnable example", expanded=False):
        st.markdown("### Runnable reference code")
        reveal_key = f"free_runnable_revealed_{prompt_key}"
        if penalty_points > 0 and not st.session_state.get(reveal_key, False):
            if st.button("Reveal runnable example", key=f"reveal_runnable_{prompt_key}"):
                st.session_state.challenge_points = float(st.session_state.get("challenge_points", 0.0)) - penalty_points
                st.session_state[reveal_key] = True
                st.warning(
                    f"Penalty applied: -{penalty_points:g} points."
                    " This will be reflected in your submission score."
                )
        if penalty_points <= 0 or st.session_state.get(reveal_key, False):
            st.markdown(_STARTER_TEMPLATE_HINT)
            st.code(runnable_code.strip(), language="python")


def _render_bonus_trivia(challenge_question: str, challenge_answer: str, answer_key: str) -> None:
    with st.expander("Bonus trivia (no penalty)", expanded=False):
        st.markdown("### Optional trivia question")
        st.caption("This is just for fun — answering correctly earns no penalty and no score impact.")
        st.markdown(challenge_question)
        bonus_answer = st.text_input("Your answer", key=answer_key)
        if bonus_answer.strip().lower() == challenge_answer.strip().lower():
            st.success("Correct!")
        elif bonus_answer.strip():
            st.warning("Not quite. Try again.")


def _render_locked_runnable(
    runnable_code: str,
    challenge_question: str,
    challenge_answer: str,
    answer_key: str,
    current_answer: str,
    penalty_points: float,
    prompt_key: str,
) -> None:
    with st.expander("Runnable example (locked)", expanded=bool(current_answer)):
        st.markdown("### Challenge unlock")
        st.markdown("Answer the challenge question correctly to reveal the runnable reference code.")
        st.markdown(challenge_question)
        user_answer = st.text_input("Enter your answer to unlock", key=answer_key)
        if user_answer.strip().lower() == challenge_answer.strip().lower():
            penalty_key = f"runnable_penalty_applied_{prompt_key}"
            if not st.session_state.get(penalty_key, False):
                st.session_state.challenge_points = float(st.session_state.get("challenge_points", 0.0)) - penalty_points
                st.session_state[penalty_key] = True
                if penalty_points > 0:
                    st.warning(
                        f"Penalty applied: -{penalty_points:g} points."
                        " This will be reflected in your submission score."
                    )
            st.success("Correct. Runnable example unlocked.")
            st.markdown(_STARTER_TEMPLATE_HINT)
            st.code(runnable_code.strip(), language="python")
        elif user_answer.strip():
            st.warning("Not quite. Try again.")


def render_prompt_help(
    example_code: str,
    runnable_code: str,
    prompt_key: str,
    challenge_question: str | None = None,
    challenge_answer: str | None = None,
    locked: bool = True,
    free_runnable_penalty: float = 0.0,
) -> None:
    answer_key = f"unlock_answer_{prompt_key}"
    current_answer = str(st.session_state.get(answer_key, "")).strip()
    penalty_points = float(CHALLENGE_PENALTIES.get(prompt_key, 3.0))

    with st.expander("Pseudo-code", expanded=False):
        st.markdown("### Pseudo-code guidance")
        st.markdown("Use this as a simple recipe. Read one line, write one matching Python line.")
        st.code(example_code.strip(), language="python")

    has_challenge = bool(challenge_question) and bool(challenge_answer)

    if not locked:
        _render_free_runnable(runnable_code, prompt_key=prompt_key, penalty_points=free_runnable_penalty)
        if has_challenge:
            _render_bonus_trivia(challenge_question, challenge_answer, answer_key)  # type: ignore[arg-type]
    elif has_challenge:
        _render_locked_runnable(runnable_code, challenge_question, challenge_answer, answer_key, current_answer, penalty_points, prompt_key)  # type: ignore[arg-type]
    else:
        _render_free_runnable(runnable_code, prompt_key=prompt_key)


def render_code_runner(
    label: str,
    key: str,
    df: pd.DataFrame,
    perf: pd.DataFrame,
    save_result_key: str | None = None,
) -> None:
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

        _render_code_runner_output(result=result, save_result_key=save_result_key)


def _render_code_runner_output(result: dict[str, object], save_result_key: str | None) -> None:
    if result["ok"]:
        st.success("Code ran successfully.")
    else:
        st.error("Code failed. See error details below.")

    stdout_text = str(result["stdout"]).strip()
    if stdout_text:
        st.markdown("Output")
        st.code(stdout_text, language="text")

    result_obj = result["result"]
    if save_result_key is not None and result["ok"]:
        st.session_state[save_result_key] = result_obj
    if isinstance(result_obj, pd.DataFrame):
        st.markdown("Result variable")
        st.dataframe(result_obj, use_container_width=True)
    elif result_obj is not None:
        st.markdown("Result variable")
        st.write(result_obj)

    error_text = str(result["error"]).strip()
    if error_text:
        st.code(error_text, language="text")


def render_prompt6_visuals(metrics_df: pd.DataFrame) -> None:
    required_cols = {"Model", "Train_RMSE", "Test_RMSE"}
    if metrics_df.empty or not required_cols.issubset(set(metrics_df.columns)):
        st.warning(
            "Need a non-empty result DataFrame with columns: Model, Train_RMSE, Test_RMSE."
        )
        return

    plot_df = metrics_df.copy()
    plot_df["Train_RMSE"] = pd.to_numeric(plot_df["Train_RMSE"], errors="coerce")
    plot_df["Test_RMSE"] = pd.to_numeric(plot_df["Test_RMSE"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Train_RMSE", "Test_RMSE", "Model"]).copy()
    if plot_df.empty:
        st.warning("No valid numeric RMSE values found to chart.")
        return

    plot_df = plot_df.sort_values("Test_RMSE", ascending=True).reset_index(drop=True)
    winner = str(plot_df.iloc[0]["Model"])
    winner_test_rmse = float(plot_df.iloc[0]["Test_RMSE"])
    winner_gap = float(plot_df.iloc[0]["Test_RMSE"] - plot_df.iloc[0]["Train_RMSE"])
    second_best_rmse = float(plot_df.iloc[1]["Test_RMSE"]) if len(plot_df) > 1 else np.nan
    lead_pct = (
        ((second_best_rmse - winner_test_rmse) / second_best_rmse) * 100
        if not np.isnan(second_best_rmse) and second_best_rmse > 0
        else np.nan
    )

    st.markdown("### Model Visual Dashboard")
    st.caption(
        "Use this panel after running Prompt 6 code. Start with winner + RMSE cards, "
        "then read the two charts to check generalization risk."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Top model", winner)
    c2.metric("Winner Test_RMSE", f"{winner_test_rmse:.2f}")
    if np.isnan(lead_pct):
        c3.metric("Lead over #2", "N/A")
    else:
        c3.metric("Lead over #2", f"{lead_pct:.1f}%")

    c4, c5 = st.columns(2)
    c4.metric("Winner train-test gap", f"{winner_gap:+.2f}")
    if winner_gap > 3.0:
        risk_level = "High"
    elif winner_gap > 1.5:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    risk_color = {
        "Low": "#22c55e",
        "Moderate": "#f59e0b",
        "High": "#ef4444",
    }[risk_level]
    c5.metric(
        "Overfitting risk",
        risk_level,
    )

    st.markdown(
        (
            "<div style='margin-top:6px;'>"
            "<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
            f"background:{risk_color};color:#ffffff;font-weight:600;'>"
            f"Overfitting Risk: {risk_level}</span>"
            "<span style='margin-left:10px;color:#d1d5db;'>"
            "Train-Test gap guide: Low (&lt;= 1.5), Moderate (1.5 to 3.0), High (&gt; 3.0)"
            "</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.success(f"Current winner by Test_RMSE: {winner}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="Test_RMSE", y="Model", palette="crest", ax=ax1)
    ax1.set_title("Leaderboard: Test RMSE by Model (Lower is Better)")
    ax1.set_xlabel("Test RMSE")
    ax1.set_ylabel("Model")
    st.pyplot(fig1)
    plt.close(fig1)

    gap_df = plot_df[["Model", "Train_RMSE", "Test_RMSE"]].melt(
        id_vars="Model",
        var_name="Split",
        value_name="RMSE",
    )
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=gap_df, x="Model", y="RMSE", hue="Split", palette="Set2", ax=ax2)
    ax2.set_title("Generalization Check: Train vs Test RMSE")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("RMSE")
    ax2.tick_params(axis="x", rotation=20)
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("See numeric table", expanded=False):
        st.dataframe(
            plot_df[["Model", "Train_RMSE", "Test_RMSE"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Interpretation quick-check")
    st.markdown("- Which model has the lowest Test_RMSE?")
    st.markdown("- Is any model's Train_RMSE much lower than Test_RMSE (overfitting warning)?")
    st.markdown("- Would you trust the winning model on new shows, and why?")


def compute_submission_score(
    metrics_df: pd.DataFrame | None,
    challenge_points: float,
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

    challenge_points = float(st.session_state.get("challenge_points", 0.0))
    score_breakdown = compute_submission_score(metrics_df=metrics_df, challenge_points=challenge_points)

    tally_rows: list[dict[str, object]] = []
    running_tally = 0.0
    for prompt_id in CHALLENGE_PENALTIES:
        locked_unlock_key = f"runnable_penalty_applied_{prompt_id}"
        free_reveal_key = f"free_runnable_revealed_{prompt_id}"
        unlocked = bool(st.session_state.get(locked_unlock_key, False) or st.session_state.get(free_reveal_key, False))
        penalty_value = float(CHALLENGE_PENALTIES[prompt_id])
        delta_points = -penalty_value if unlocked else 0.0
        running_tally += delta_points
        tally_rows.append(
            {
                "prompt": prompt_id,
                "runnable_unlocked": unlocked,
                "penalty_points": round(penalty_value, 2),
                "points_applied": round(delta_points, 2),
            }
        )

    unlock_tally_json = json.dumps(tally_rows)

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
        "challenge_points": round(challenge_points, 2),
        "challenge_tally_total": round(running_tally, 2),
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
        "challenge_unlock_tally_json": unlock_tally_json,
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
    st.markdown("### Unlock and penalty tally")
    st.caption("This shows where runnable examples were unlocked and how each unlock affected points.")
    st.dataframe(pd.DataFrame(tally_rows), use_container_width=True)
    st.info(
        f"Total challenge adjustment: {round(challenge_points, 2):+g} points"
    )
    st.dataframe(new_row_df, use_container_width=True)


if "holdout_metrics" not in st.session_state:
    st.session_state.holdout_metrics = None
if "holdout_models" not in st.session_state:
    st.session_state.holdout_models = {}
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "challenge_points" not in st.session_state:
    st.session_state.challenge_points = 0.0
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

    intro_col_1, intro_col_2 = st.columns(2)
    with intro_col_1:
        st.markdown("### Setlist planning roadmap")
        st.markdown("1. Decide what Ben is trying to predict after each show.")
        st.markdown("2. Label that prediction target clearly.")
        st.markdown("3. Treat the remaining columns like clues the band can see before taking the stage.")
        st.markdown("4. Build a quick summary before moving into model prep.")
    with intro_col_2:
        st.markdown("### What Ben needs from this step")
        st.markdown("- One target column: `SCORE`.")
        st.markdown("- A list of candidate features the model can learn from.")
        st.markdown("- A short summary showing the target and example features.")

    with st.expander("Why this matters for the band", expanded=False):
        st.markdown("- The target is the thing Ben wants to predict after a future gig.")
        st.markdown("- Features are the signals available before the show starts.")
        st.markdown("- If you mix up the target and the features, the model will rehearse the wrong song.")
        st.markdown("- This step tells the model what to listen to before Prompt 5 builds the rehearsal data.")

    st.markdown("### Suggested backstage order")
    st.caption("Treat this like assigning roles before the band walks on stage.")
    st.markdown("1. Set `target_column = 'SCORE'`.")
    st.markdown("2. Build `feature_columns` from every other remaining column.")
    st.markdown("3. Count how many features are left after cleanup.")
    st.markdown("4. Save a simple summary in `result`.")

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe"
            "\n# 1. Assume Prompt 3 already removed the bad columns"
            "\n# 2. Defensive move: Make a copy to avoid changing the original"
            "\n#    Python idea: model_df = df.copy()"
            "\n# 3. Identify the prediction target explicitly"
            "\n#    Python idea: target_column = 'SCORE' (this is what Ben wants to predict)"
            "\n# 4. Build y from the target column"
            "\n#    Python idea: y = model_df[target_column].values"
            "\n# 5. Build X from all columns except the target"
            "\n#    Python idea: X = model_df.drop(columns=[target_column]).values"
            "\n# 6. Save a quick summary in result (no splitting yet—that happens in Prompt 5)"
            "\n#    Python idea: result = {'target_column': target_column, 'X_shape': X.shape, 'y_shape': y.shape, 'feature_count': X.shape[1]}"
            "\n# 7. Print the result so you know what you are working with"
            "\n#    Python idea: print(result)"
        ),
        runnable_code=(
            "# Step 1: Assume df is already cleaned from Prompt 3"
            "\nclean_df = df.copy()"
            "\n"
            "\n# Main approach: explicit target selection (safer than positional selection)"
            "\nmodel_df = clean_df.copy()"
            "\n"
            "\n# SCORE is our specific prediction target (set by the band's goal)"
            "\ntarget_column = 'SCORE'"
            "\ny = model_df[target_column].values"
            "\nX = model_df.drop(columns=[target_column]).values"
            "\n"
            "\nresult = {'target_column': target_column, 'X_shape': X.shape, 'y_shape': y.shape, 'feature_count': X.shape[1]}"
            "\nprint(result)"
            "\n"
        ),
        prompt_key="p4_shape",
        challenge_question=p4_challenge_question,
        challenge_answer=p4_challenge_answer,
    )
    render_code_runner(
        label="Write your code for Prompt 4 (choose the band's target and signals)",
        key="p4_shape",
        df=df,
        perf=perf,
    )

    st.markdown("### Backstage checklist")
    st.markdown("- Did you set `SCORE` as the target?")
    st.markdown("- Did you exclude the target from the feature list?")

with st.expander("Prompt 5: Prepare model-ready data", expanded=False):
    st.info("Story beat: before the rehearsal lab starts, prepare clean modeling inputs and a reliable train/test split.")

    intro_col_1, intro_col_2 = st.columns(2)
    with intro_col_1:
        st.markdown("### Soundcheck roadmap")
        st.markdown("1. Bring X and y from Prompt 4 (target and features already defined).")
        st.markdown("2. Split the gigs into rehearsal data (train) and live-show test data.")
        st.markdown("3. Verify the split so you know both sets have enough gigs to learn from.")
    with intro_col_2:
        st.markdown("### What Ben needs from you")
        st.markdown("- `X_train`, `X_test`, `y_train`, and `y_test`.")
        st.markdown("- A shape summary showing how the split divided the gigs.")
        st.markdown("- Confidence that train and test sizes are reasonable before Prompt 6.")

    with st.expander("Why this step matters for the band", expanded=False):
        st.markdown("- Prompt 4 already told us what to predict (SCORE) and what features to use (X).")
        st.markdown("- Now we split the gigs into **practice shows** (train) and **unseen shows** (test).")
        st.markdown("- The model learns patterns from training data, but we test it on completely new gigs.")
        st.markdown("- This is the only fair way to know if the model will work on future shows.")
        st.markdown("- If we trained and tested on the same data, the model would cheat and look better than it really is.")
    st.markdown("### Suggested setup order")
    st.caption("Build this like a pre-show setup: one cable at a time.")
    st.markdown("1. Assume X and y are already prepared from Prompt 4.")
    st.markdown("2. Use `train_test_split(...)` to divide the gigs into practice (train) and test sets.")
    st.markdown("3. Print the shapes so you know the split worked before Prompt 6 begins.")

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe\n"
            "# Step 1: Assume X and y are already prepared from Prompt 4\n"
            "# Step 2: Split the data into rehearsal data and live-show test data\n"
            "#    Python idea: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=...)\n"
            "#    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html\n"
            "# Step 3: Save quick shape checks in result to verify the split worked\n"
            "# Step 4: Print the shapes before moving to Prompt 6\n"
        ),
        runnable_code=(
            "# Step 1: Split the data into rehearsal data and live-show test data\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            "\n"
            "# Step 2: Save quick shape checks in result to verify the split worked\n"
            "result = {'x_train_shape': X_train.shape, 'x_test_shape': X_test.shape, 'y_train_shape': y_train.shape, 'y_test_shape': y_test.shape}\n"
            "\n"
            "# Step 3: Print the shapes before moving to Prompt 6\n"
            "print(result)\n"
        ),
        prompt_key="p5_prepare",
        challenge_question=p5_challenge_question,
        challenge_answer=p5_challenge_answer,
        locked=True,
    )
    render_code_runner(
        label="Write your code for Prompt 5 (prep the band's rehearsal data)",
        key="p5_prepare",
        df=df,
        perf=perf,
    )


with st.expander("Prompt 6: Train and compare regression models", expanded=False):
    st.info("Story beat: run the rehearsal benchmark. Train three regressors and compare fairly.")

    intro_col_1, intro_col_2 = st.columns(2)
    with intro_col_1:
        st.markdown("### Rehearsal roadmap")
        st.markdown("1. Bring X_train, X_test, y_train, y_test from Prompt 5 (split data ready to use).")
        st.markdown("2. Train three models on the training data.")
        st.markdown("3. Evaluate each model on train and test data.")
        st.markdown("4. Sort by `Test_RMSE` to see which model reads the crowd best.")
    with intro_col_2:
        st.markdown("### What to hand Ben")
        st.markdown("- A final DataFrame named `result`.")
        st.markdown("- 3 total rows: one per trained model.")
        st.markdown("- Columns for Train/Test MAE, RMSE, and R2.")
        st.markdown("- Lowest `Test_RMSE` at the top, so Ben can quickly see the winner.")

    with st.expander("What do these models do?", expanded=False):
        st.markdown("**LinearRegression:** Finds a straight-line pattern. It learns: 'Higher team size = higher score' or 'Faster songs = lower scores.' Simple and easy to explain.")
        st.markdown("**RandomForestRegressor:** Like asking 100 different judges and averaging their votes. It can catch weird patterns LinearRegression misses.")
        st.markdown("**KNeighborsRegressor:** Looks backward in time: 'Your next show is like shows #5, #12, and #18, so I'll guess the average of those.' Learns from similarity.")
        st.markdown("\n**Train metrics:** How well the model did on past shows (it saw these before).")
        st.markdown("**Test metrics:** How well it guesses *new* shows (it's never seen these). This is the real test.")

    st.markdown("### Suggested soundcheck order")
    st.caption("Build this one step at a time. Start with one model, then add the next two.")
    st.markdown("**Step 1:** Load X_train, X_test, y_train, y_test from Prompt 5.")
    st.markdown("**Step 2:** Create a list called `rows` to store results.")
    st.markdown("**Step 3:** For each of the 3 models, fit it on train data, predict on both train and test, calculate metrics, and save.")
    st.markdown("**Step 4:** Convert `rows` to a DataFrame and sort by `Test_RMSE` (lower is better on unseen shows).")
    st.markdown("**Pro tip:** Run and test each step before moving to the next. Don't try to write everything at once.")

    p6_runnable_code = (
        "# STEP 1: Assume X_train, X_test, y_train, y_test are ready from Prompt 5\n"
        "# (If doing this standalone, you can rebuild from perf as shown in Prompt 5 runnable)\n"
        "\n"
        "# STEP 2: Create a list to store all model results\n"
        "rows = []\n"
        "\n"
        "# STEP 3: List the 3 models to train\n"
        "models = [\n"
        "    ('LinearRegression', LinearRegression()),\n"
        "    ('RandomForestRegressor', RandomForestRegressor(n_estimators=100, random_state=42)),\n"
        "    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=5))\n"
        "]\n"
        "\n"
        "# STEP 4: Train each model and evaluate on train and test\n"
        "for name, model in models:\n"
        "    model.fit(X_train, y_train)  # Fit (train) on past shows\n"
        "    pred_train = model.predict(X_train)  # Predict on past shows\n"
        "    pred_test = model.predict(X_test)  # Predict on new shows (this is the real test)\n"
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
        "# STEP 5: Show results sorted by test performance (Test_RMSE lowest = best on new shows)\n"
        "result = pd.DataFrame(rows).sort_values('Test_RMSE', ascending=True).reset_index(drop=True)\n"
        "print(result)\n"
    )

    p6_starter_scaffold = (
        "# LIGHT STARTER: begin with ONE model, then add the other two\n"
        "rows = []\n"
        "models = [('LinearRegression', LinearRegression())]\n"
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
        "result = pd.DataFrame(rows).sort_values('Test_RMSE').reset_index(drop=True)\n"
        "print(result)\n"
        "\n"
        "# Challenge extension: add RandomForestRegressor and KNeighborsRegressor to models\n"
    )

    render_prompt_help(
        example_code=(
            "# Pseudo-code recipe\n"
            "# 1. Load X_train, X_test, y_train, y_test from Prompt 5\n"
            "# 2. Create an empty list: rows = []\n"
            "# 3. Create a list of 3 models to train\n"
            "# 4. FOR each model:\n"
            "#    a. Fit it on X_train, y_train\n"
            "#    b. Predict on X_train and X_test\n"
            "#    c. Calculate metrics (MAE, RMSE, R2) for both\n"
            "#    d. Add results to rows\n"
            "# 5. Convert rows to DataFrame\n"
            "# 6. Sort by Test_RMSE (best on unseen data is at top)\n"
            "# 7. Save as result\n"
        ),
        runnable_code=p6_runnable_code,
        prompt_key="p6_modeling",
        challenge_question=p6_challenge_question,
        challenge_answer=p6_challenge_answer,
    )

    beginner_mode = st.toggle(
        "Beginner mode for Prompt 6",
        value=True,
        key="p6_beginner_mode",
        help="ON = one-model starter first. OFF = full 3-model reference starter.",
    )
    if beginner_mode:
        st.caption("Beginner mode is ON. Starter code begins with one model, then you extend it.")
    else:
        st.caption("Advanced mode is ON. Starter code includes the full 3-model workflow.")

    if st.button("Insert starter into Prompt 6 editor", key="insert_p6_template_code"):
        if beginner_mode:
            st.session_state["prompt_code_p6_modeling"] = p6_starter_scaffold
            st.success("Beginner starter inserted.")
        else:
            st.session_state["prompt_code_p6_modeling"] = p6_runnable_code
            st.success("Advanced starter inserted.")

    with st.expander("Optional: insert the other starter version", expanded=False):
        c_insert_1, c_insert_2 = st.columns(2)
        if c_insert_1.button("Insert beginner starter", key="insert_p6_beginner_alt"):
            st.session_state["prompt_code_p6_modeling"] = p6_starter_scaffold
            st.success("Beginner starter inserted.")
        if c_insert_2.button("Insert advanced starter", key="insert_p6_advanced_alt"):
            st.session_state["prompt_code_p6_modeling"] = p6_runnable_code
            st.success("Advanced starter inserted.")

    tab_build, tab_visuals, tab_interpret = st.tabs(
        ["Build", "Visualize", "Interpret"]
    )

    with tab_build:
        st.markdown("### Build Track")
        st.caption("Goal: produce a DataFrame named result with one row per model and sorted by Test_RMSE.")
        if beginner_mode:
            st.markdown("1. Start with one model first (LinearRegression).")
            st.markdown("2. Make sure your code runs and returns result.")
            st.markdown("3. Add RandomForestRegressor and KNeighborsRegressor after Step 2 works.")
        else:
            st.markdown("1. Start with all three models in your models list.")
            st.markdown("2. Run once to confirm result has 3 rows.")
            st.markdown("3. Improve readability with comments and clear variable names.")
        render_code_runner(
            label="Write your code for Prompt 6 (build the band benchmark step by step)",
            key="p6_modeling",
            df=df,
            perf=perf_for_model if 'perf_for_model' in locals() and perf_for_model is not None else df,
            save_result_key="p6_result_df",
        )

    with tab_visuals:
        st.markdown("### Visualize Track")
        st.caption("Goal: quickly identify the winner and check overfitting risk.")
        st.markdown("1. Confirm which model has the lowest Test_RMSE.")
        st.markdown("2. Check whether Train_RMSE is much lower than Test_RMSE.")
        metrics_source = st.session_state.get("p6_result_df")
        if not isinstance(metrics_source, pd.DataFrame):
            metrics_source = st.session_state.get("holdout_metrics")

        if isinstance(metrics_source, pd.DataFrame):
            render_prompt6_visuals(metrics_source)
        else:
            st.info(
                "Run your Prompt 6 code first (with a DataFrame assigned to result), "
                "or use the built-in benchmark button to generate model metrics."
            )

    with tab_interpret:
        st.markdown("### Interpret Track")
        st.caption("Goal: write short analysis notes now so Prompt 7 feels easier.")
        st.markdown("1. Winner model: <name>")
        st.markdown("2. Why it won: <lowest Test_RMSE or strongest generalization>")
        st.markdown("3. Risk level: <low / moderate / high>")
        st.markdown("4. One practical recommendation for Ben before next show")
        st.text_area(
            "Prompt 6 interpretation notes (used in Prompt 7)",
            key="p6_interpret_notes",
            height=120,
            placeholder="Example: RandomForest had the lowest Test_RMSE and moderate train-test gap, so it is the best current choice but should be validated on more gigs.",
        )

    st.markdown("### Backstage checklist before the encore")
    st.markdown("- ✅ Did you fit each model on train data and evaluate on both train and test?")
    st.markdown("- ✅ Did your result table show exactly 3 trained models?")
    st.markdown("- ✅ Did you compare train vs. test metrics to spot overfitting (train much better = risky)?")
    st.markdown("- 💡 **Next:** The model with the lowest Test_RMSE is your winner for Prompt 7.")

    st.markdown("### Built-in benchmark settings")
    st.caption("Use these to configure the optional automated benchmark button below. These do NOT affect your code.")
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
        st.warning(
            f"Could not rebuild performance table from current settings ({exc}). "
            "Using the default backend performance table so you can continue."
        )
        perf_for_model = perf.copy()

    if st.button("Use built-in benchmark and send to Prompt 7", key="p6_builtin_benchmark", disabled=('perf_for_model' not in locals() or perf_for_model is None)):
        if 'perf_for_model' in locals() and perf_for_model is not None:
            metrics_df, trained_models, split_data = evaluate_holdout(
                perf=perf_for_model,
                model_names=["LinearRegression", "RandomForestRegressor", "KNeighborsRegressor"],
                test_size=test_size_pct / 100.0,
                seed=int(random_seed),
                include_baseline=False,
            )
            st.session_state.holdout_metrics = metrics_df
            st.session_state.holdout_models = trained_models
            st.session_state.split_data = split_data
            st.success("Benchmark saved. Prompt 7 is now ready.")
            st.dataframe(metrics_df, use_container_width=True)


with st.expander("Prompt 7 (Challenging): Interpret your best model", expanded=False):
    st.info("Story beat: turn your Prompt 6 interpretation into a short manager memo. Focus on clear communication, not more coding.")

    metrics_df = st.session_state.holdout_metrics
    if metrics_df is not None and not metrics_df.empty:
        best_row = metrics_df.sort_values("Test_RMSE", ascending=True).iloc[0]
        st.success(
            f"Current best model from saved metrics: {best_row['Model']} | Test_RMSE={float(best_row['Test_RMSE']):.2f}"
        )

    p6_notes = str(st.session_state.get("p6_interpret_notes", "")).strip()
    if p6_notes:
        st.markdown("### Notes carried from Prompt 6")
        st.info(p6_notes)

    intro_col_1, intro_col_2 = st.columns(2)
    with intro_col_1:
        st.markdown("### Your job backstage")
        st.markdown("Write a short note for Ben's manager, not for the data team.")
        st.markdown("Your goal is to answer one question clearly: **Can the band trust this model before the next show?**")
    with intro_col_2:
        st.markdown("### Keep it stage-ready")
        st.markdown("- Use plain English")
        st.markdown("- Mention the winning model")
        st.markdown("- Explain what the numbers mean for future shows")
        st.markdown("- End with a practical next step for the band")

    with st.expander("How to explain this to Ben's manager", expanded=False):
        st.markdown("- Do not just list metrics. Explain what they mean for booking and planning future shows.")
        st.markdown("- `Lower Test_RMSE` means the model makes smaller mistakes on new performances.")
        st.markdown("- Be honest about limits. A useful scouting tool can still miss unusual gigs.")
        st.markdown("- Ben's manager wants to know: what won, why it matters, what risk remains, and what the band should do next.")

    st.markdown("### Build the manager memo in 4 parts")
    st.markdown("1. **Headliner:** Which model performed best on the test set?")
    st.markdown("2. **Show value:** Why is that useful for predicting future performance scores?")
    st.markdown("3. **Risk:** Does the train vs. test gap suggest overfitting or instability?")
    st.markdown("4. **Recommendation:** What should the band do next before relying on it more?")
    
    with st.expander("Prompt 7 writing guide", expanded=False):
        st.markdown("1. Open with the winning model from Prompt 6.")
        st.markdown("2. Explain in plain English what the performance means for future shows.")
        st.markdown("3. Name one risk based on train-test behavior.")
        st.markdown("4. End with one recommendation Ben can act on now.")

    if st.button("Insert writing template into Prompt 7 answer", key="insert_p7_template"):
        prompt6_notes_block = f"Prompt 6 notes: {p6_notes}\n\n" if p6_notes else ""
        st.session_state["analysis_answer"] = (
            prompt6_notes_block +
            "1) The best model was <model_name>, based on the lowest Test_RMSE.\n"
            "2) Compared with the next-best model, it improved prediction error by <value>%, which means <plain English meaning for future shows>.\n"
            "3) The train vs. test results suggest <low / moderate / high> overfitting risk.\n"
            "4) One limitation is that the model may struggle when <type of performance or concert condition>.\n"
            "5) My recommendation for Ben's team is to <one practical next step>."
        )
        st.success("Prompt 7 writing template inserted.")

    st.caption("Sentence starters: The best model was... | For Ben's next shows, this means... | One risk is... | My recommendation for the band is...")
    
    st.markdown("### Your interpretation")
    st.text_area(
        "Write a short manager-friendly note for Ben's band.",
        key="analysis_answer",
        height=150,
        placeholder="Example: RandomForest was the strongest model because it had the lowest Test RMSE. "
                    "Compared with the next-best model, it made meaningfully smaller prediction errors on unseen gigs, which makes it a stronger guide for future set planning. "
                    "That suggests it could help Ben prepare for upcoming shows, but the train/test gap shows we should still watch for overfitting. "
                    "It may struggle on unusual performances that do not sound like the past data. "
                    "Before using it more broadly, I would validate it on more gigs and tune the model settings.",
    )

    st.markdown("### Final soundcheck")
    st.markdown("- Did you explain the winner in plain English, not just with model jargon?")
    st.markdown("- Did you mention what the result means for Ben's next shows?")
    st.markdown("- Did you identify one risk or limitation?")
    st.markdown("- Did you end with one practical recommendation for the band?")

st.divider()
render_team_results_store(
    remove_outliers=remove_outliers,
    test_size_pct=int(test_size_pct),
    random_seed=int(random_seed),
)

st.caption("Intern Game Day 2026 | Ben's Rock Band")
