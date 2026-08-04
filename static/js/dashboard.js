const totalVisitsElement =
    document.getElementById("total-visits");

const averageDurationElement =
    document.getElementById("average-duration");

const maleVisitsElement =
    document.getElementById("male-visits");

const femaleVisitsElement =
    document.getElementById("female-visits");

const unknownGenderVisitsElement =
    document.getElementById("unknown-gender-visits");

const ageGroupsElement =
    document.getElementById("age-groups");

const connectionStatusElement =
    document.getElementById("connection-status");


function renderAgeGroups(ageGroups) {
    ageGroupsElement.innerHTML = "";

    const entries = Object.entries(ageGroups);

    if (entries.length === 0) {
        const emptyMessage = document.createElement("p");

        emptyMessage.className = "empty-message";
        emptyMessage.textContent = "No completed visits yet.";

        ageGroupsElement.appendChild(emptyMessage);
        return;
    }

    for (const [ageGroup, count] of entries) {
        const row = document.createElement("div");
        row.className = "age-row";

        const label = document.createElement("span");
        label.textContent = ageGroup;

        const value = document.createElement("strong");
        value.textContent = count;

        row.appendChild(label);
        row.appendChild(value);

        ageGroupsElement.appendChild(row);
    }
}


function renderStatistics(statistics) {
    totalVisitsElement.textContent =
        statistics.total_visits;

    averageDurationElement.textContent =
        `${statistics.average_duration.toFixed(1)} s`;

    maleVisitsElement.textContent =
        statistics.male_visits;

    femaleVisitsElement.textContent =
        statistics.female_visits;

    unknownGenderVisitsElement.textContent =
        statistics.unknown_gender_visits;

    renderAgeGroups(statistics.age_groups);
}


async function loadStatistics() {
    try {
        const response = await fetch(
            "/api/statistics",
            {
                cache: "no-store",
            }
        );

        if (!response.ok) {
            throw new Error(
                `Statistics request failed: ${response.status}`
            );
        }

        const statistics = await response.json();

        renderStatistics(statistics);
    } catch (error) {
        console.error("Could not load statistics:", error);
    }
}


// Load existing SQLite statistics when the page first opens.
loadStatistics();


// Keep one connection open for database-update notifications.
const statisticsStream =
    new EventSource("/api/statistics-stream");


statisticsStream.onopen = function () {
    connectionStatusElement.textContent = "Live";
    connectionStatusElement.classList.add("connected");
};


statisticsStream.onmessage = async function (event) {
    const message = JSON.parse(event.data);

    if (message.type === "statistics_updated") {
        // SQLite has changed, so request fresh calculated statistics.
        await loadStatistics();
    }
};


statisticsStream.onerror = function () {
    connectionStatusElement.textContent = "Reconnecting";
    connectionStatusElement.classList.remove("connected");

    // EventSource reconnects automatically.
};