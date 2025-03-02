async function sendQuery() {
    const queryText = document.getElementById("query").value;
    const responseElement = document.getElementById("response");

    if (!queryText.trim()) {
        responseElement.innerText = "Ecrit quelque chose.";
        return;
    }

    responseElement.innerText = "Hmmmmm";

    try {
        const response = await fetch("http://localhost:8000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query_text: queryText })
        });

        const data = await response.json();
        responseElement.innerText = data.response || "Aucune réponse trouvée.";
    } catch (error) {
        responseElement.innerText = "Pb de connexion serveur";
        console.error("Erreur :", error);
    }
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendQuery();
    }
}
