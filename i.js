  const ujjainCoords = [23.1765, 75.7885];

  // Prediction Map
  const predictedMap = L.map("predicted-map").setView(ujjainCoords, 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors"
  }).addTo(predictedMap);

  // Current Situation Map
  const currentMap = L.map("current-map").setView(ujjainCoords, 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors"
  }).addTo(currentMap);

  // Optional markers
  L.marker(ujjainCoords).addTo(predictedMap).bindPopup("Prediction - Ujjain").openPopup();
  L.marker(ujjainCoords).addTo(currentMap).bindPopup("Current Situation - Ujjain").openPopup();




const submitBtn = document.getElementById('submitBtn');
const summaryList = document.getElementById('summaryList');

submitBtn.addEventListener('click', () => {
  const eventType = document.getElementById('eventType').value;
  const eventDate = document.getElementById('eventDate').value;
  const eventTime = document.getElementById('eventTime').value;
  const peakHour = document.getElementById('peakHour').value;
  const rain = document.getElementById('rainExpected').value;

  alert(`Event: ${eventType}\nDate: ${eventDate}\nTime: ${eventTime}\nPeak: ${peakHour}\nRain: ${rain}`);
  
  // Example: Update summary dynamically
  summaryList.innerHTML = `
    <li>Event: ${eventType}</li>
    <li>Date & Time: ${eventDate} ${eventTime}</li>
    <li>Peak Status: ${peakHour}</li>
    <li>Weather: ${rain}</li>
    <li>Deploy security and coordinate traffic accordingly</li>
  `;
});



// Example: Backend data
const predictedPoints = [
  "Deploy security in high-density areas",
  "Redirect crowd to alternate routes",
  "Setup emergency medical aid points",
  "Monitor entry points",
  "Coordinate with traffic control"
];

const currentPoints = [
  "Monitor crowd movement in real-time",
  "Broadcast announcements via speakers",
  "Check emergency medical aid points",
  "Report congestion areas",
  "Update public transport coordination"
];

// Function to display points
function displaySummary() {
  const predictedList = document.getElementById('predictedList');
  const currentList = document.getElementById('currentList');

  predictedList.innerHTML = predictedPoints.map(point => `<li>${point}</li>`).join('');
  currentList.innerHTML = currentPoints.map(point => `<li>${point}</li>`).join('');
}

// Call on page load
window.onload = displaySummary;











  // const statusColor = {
  //   "normal": "green",
  //   "medium": "yellow",
  //   "high": "red"
  // };

  // // Data from backend Flask
  // const zoneStatus = {{ zone_status | tojson }};

  // // Update indicator colors
  // for (const [zoneId, status] of Object.entries(zoneStatus)) {
  //   const el = document.getElementById(zoneId);
  //   if(el) el.style.backgroundColor = statusColor[status] || "green";
  // }

