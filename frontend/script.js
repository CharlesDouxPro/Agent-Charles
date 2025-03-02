async function sendQuery() {
    const queryText = document.getElementById("query").value;
    const responseElement = document.getElementById("response");

    if (!queryText.trim()) {
        responseElement.innerText = "❌ Merci d'écrire une question.";
        return;
    }

    responseElement.innerText = "⏳ Génération en cours...";

    try {
        const response = await fetch("http://localhost:8000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query_text: queryText })
        });

        const data = await response.json();
        responseElement.innerText = data.response || "❌ Aucune réponse trouvée.";
    } catch (error) {
        responseElement.innerText = "⚠️ Erreur de connexion au serveur.";
        console.error("Erreur :", error);
    }
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendQuery();
    }
}
