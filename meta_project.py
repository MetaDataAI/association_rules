import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from mlxtend.frequent_patterns import apriori, association_rules
from flask import Flask, render_template, request
import networkx as nx
import seaborn as sns

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"  # Diretório onde o gráfico será salvo
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form.get("model")
        min_support = float(request.form.get("min_support"))
        min_confidence = float(request.form.get("min_confidence"))
        lift = float(request.form.get("lift"))
        file = request.files.get("dataset")

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Process dataset
            try:
                df = pd.read_csv(filepath)
                rules, chart_data = process_dataset(df, min_support, min_confidence, lift)
            except Exception as e:
                return f"<h1>Error processing dataset: {e}</h1>"

            return render_template(
                "result.html",
                model=model,
                min_support=min_support,
                min_confidence=min_confidence,
                lift=lift,
                filename=file.filename,
                rules=rules.to_html(classes="table table-striped", index=False),
                chart_data=chart_data  # Enviar gráficos como base64
            )

    return render_template("index.html")


def process_dataset(df, min_support, min_confidence, lift):
    # Verifica se o dataset está no formato transacional
    if "Purchased Items" in df.columns:
        # Remove espaços e separa os itens da transação
        df["Purchased Items"] = df["Purchased Items"].apply(lambda x: x.replace(" ", "").split(","))
        
        # Converte para o formato one-hot encoding
        transactions = df["Purchased Items"].tolist()
        unique_items = list(set(item for transaction in transactions for item in transaction))  # Converte set para lista
        
        # Cria o DataFrame de one-hot encoding
        one_hot_encoded = pd.DataFrame(
            [[1 if item in transaction else 0 for item in unique_items] for transaction in transactions],
            columns=unique_items  # Agora é uma lista, não um set
        )
        df = one_hot_encoded
    else:
        # Verifica se o DataFrame já está no formato binário
        if not all(df.isin([0, 1]).all()):
            raise ValueError("Dataset must contain only binary values (0 or 1) in one-hot encoding format.")

    # Aplicar Apriori
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    num_itemsets = len(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)
    rules = rules[rules["lift"] >= lift]

    # Gerar gráficos
    chart_data = []
    chart_data.append(generate_bar_chart(frequent_itemsets))
    chart_data.append(generate_line_chart(frequent_itemsets))
    chart_data.append(generate_scatter_plot(rules))
    chart_data.append(generate_network_graph(rules))
    chart_data.append(generate_pie_chart(df))
    chart_data.append(generate_histogram(rules))
    chart_data.append(generate_heatmap(rules))
    chart_data.append(generate_pareto_chart(frequent_itemsets))
    chart_data.append(generate_box_plot(rules))

    return rules, chart_data


# Funções para gerar os gráficos

def generate_bar_chart(frequent_itemsets):
    fig, ax = plt.subplots(figsize=(10, 6))
    frequent_itemsets.nlargest(10, "support").plot(kind="bar", x="itemsets", y="support", ax=ax, legend=False)
    ax.set_title("Top 10 Frequent Itemsets")
    ax.set_ylabel("Support")
    return save_chart_as_base64(fig)

def generate_line_chart(frequent_itemsets):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequent_itemsets["support"], color="blue", label="Support")
    ax.set_title("Support Over Time")
    ax.set_ylabel("Support")
    return save_chart_as_base64(fig)

def generate_scatter_plot(rules):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(rules["support"], rules["confidence"], alpha=0.5)
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Scatter Plot of Support vs Confidence")
    return save_chart_as_base64(fig)

def generate_network_graph(rules):
    # Verificar se temos antecedents e consequents
    G = nx.Graph()
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        for antecedent in antecedents:
            for consequent in consequents:
                G.add_edge(antecedent, consequent)

    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", font_size=10, ax=ax)
    ax.set_title("Network Graph of Item Relationships")
    return save_chart_as_base64(fig)

def generate_pie_chart(df):
    item_counts = df.sum()  # Contar o número de vezes que cada item aparece
    fig, ax = plt.subplots(figsize=(7, 7))
    item_counts.plot(kind='pie', autopct='%1.1f%%', legend=False, ax=ax)
    ax.set_title("Item Purchase Distribution")
    return save_chart_as_base64(fig)

def generate_histogram(rules):
    fig, ax = plt.subplots(figsize=(10, 6))
    rules["support"].hist(bins=20, color="skyblue", edgecolor="black", ax=ax)
    ax.set_title("Histogram of Support Values")
    ax.set_xlabel("Support")
    ax.set_ylabel("Frequency")
    return save_chart_as_base64(fig)

def generate_heatmap(rules):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = rules[["support", "confidence", "lift"]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap of Support, Confidence, and Lift")
    return save_chart_as_base64(fig)

def generate_pareto_chart(frequent_itemsets):
    fig, ax = plt.subplots(figsize=(10, 6))
    frequent_itemsets['support'].nlargest(10).plot(kind='bar', color='skyblue', ax=ax)
    cumulative_support = frequent_itemsets['support'].nlargest(10).cumsum()
    ax.plot(cumulative_support, color='red', marker='o', linestyle='--', label='Cumulative Support')
    ax.set_title("Pareto Chart of Frequent Itemsets")
    return save_chart_as_base64(fig)

def generate_box_plot(rules):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=rules["support"], color="lightblue", ax=ax)
    ax.set_title("Box Plot of Support")
    return save_chart_as_base64(fig)


def save_chart_as_base64(fig):
    """Converte um gráfico em Base64 para exibição em HTML"""
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    img.seek(0)
    chart_data = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close()
    return chart_data


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    app.run(debug=True)
