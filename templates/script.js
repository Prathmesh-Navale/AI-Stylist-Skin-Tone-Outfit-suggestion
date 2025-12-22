let outfitsData = {};
let currentGender = 'male';
let currentCategory = 'Bussiness_attire';
let favorites = JSON.parse(localStorage.getItem("favorites")) || [];

// Switch Gender
function switchGender(gender) {
  currentGender = gender;
  document.querySelectorAll('#gender-tabs .tab').forEach(tab => tab.classList.remove('active'));
  document.querySelector(`#gender-tabs .tab:nth-child(${gender === 'male' ? 1 : 2})`).classList.add('active');
  renderImages();
}

// Switch Category
function switchCategory(category) {
  currentCategory = category;
  document.querySelectorAll('#category-tabs .tab').forEach(tab => tab.classList.remove('active'));
  const indexMap = { 'Bussiness_attire':1, 'Formal_wear':2, 'Kurti':3, 'Wedding':4 };
  document.querySelector(`#category-tabs .tab:nth-child(${indexMap[category]})`).classList.add('active');
  renderImages();
}

// Render Outfits
function renderImages() {
  const grid = document.getElementById('image-grid');
  grid.innerHTML = '';
  if (outfitsData[currentGender] && outfitsData[currentGender][currentCategory]) {
    const shoppingLinks = {
      'Bussiness_attire': "https://www.amazon.in/s?k=business+attire",
      'Formal_wear': "https://www.myntra.com/formal-wear",
      'Kurti': "https://www.flipkart.com/search?q=kurti",
      'Wedding': "https://www.myntra.com/wedding-dresses"
    };

    outfitsData[currentGender][currentCategory].forEach(img => {
      grid.innerHTML += `
        <div class="card">
          <img src="/static/${img}" alt="${currentGender} ${currentCategory} Outfit">
          <p>${currentCategory.replace('_',' ')}</p>
          <button class="save-btn" onclick="saveFavorite('${img}')">❤️ Save</button>
          <a href="${shoppingLinks[currentCategory]}" target="_blank">
            <button class="save-btn" style="background:linear-gradient(45deg,#28a745,#218838);">🛒 Buy Now</button>
          </a>
        </div>`;
    });
  }
}

// Detect Skin Tone
async function detect() {
  const res = await fetch('/detect', { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify({}) 
  });
  const data = await res.json();
  outfitsData = { male: data.male_outfits, female: data.female_outfits };
  document.getElementById('result').innerHTML = `<strong>Detected Skin Tone:</strong> ${data.skin_tone}`;
  document.getElementById('selection-section').style.display = "block";
  renderImages();
}

// Save to Favorites
function saveFavorite(imgPath) {
  if (!favorites.includes(imgPath)) {
    favorites.push(imgPath);
    localStorage.setItem("favorites", JSON.stringify(favorites));
    renderFavorites();
  } else {
    alert("⚠️ Already in Favorites!");
  }
}

// Remove from Favorites
function removeFavorite(imgPath) {
  favorites = favorites.filter(fav => fav !== imgPath);
  localStorage.setItem("favorites", JSON.stringify(favorites));
  renderFavorites();
}

// Render Favorites
function renderFavorites() {
  const favSection = document.getElementById("favorites-grid");
  favSection.innerHTML = '';
  favorites.forEach(img => {
    favSection.innerHTML += `
      <div class="card">
        <img src="/static/${img}" alt="Favorite Outfit">
        <p>Saved Outfit</p>
        <button class="remove-btn" onclick="removeFavorite('${img}')">❌ Remove</button>
      </div>`;
  });
}
renderFavorites();
