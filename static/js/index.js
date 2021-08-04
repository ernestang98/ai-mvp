// Not sure if lines 3 - 6 is needed for production

let baseURl = "SET_BASE_URL_HERE"

if (location.hostname === "localhost" || location.hostname === "0.0.0.0" || location.hostname === "127.0.0.1")
    baseURl = ""

function recalibrate(toReset = "SPIT") {
    fetch("/reset", {
        method: 'POST',
        mode: 'cors',
        cache: 'no-cache',
        credentials: 'same-origin',
        headers: {'Content-Type': 'application/json'},
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
        body: JSON.stringify({"type": toReset})
    }).then(response => response.json())
        .then(data => console.log(data))
        .catch(err => console.log(err))
}

const img = document.getElementById('cv')
const imgLoading = document.getElementById('loading')

const imgText = document.getElementById('cv-text')
const imgLoadingText = document.getElementById('loading-text')

function loaded() {
    alert('loaded')
    imgLoading.style.display = "none"
    img.style.display = "block"
    imgLoadingText.style.display = "none"
    imgText.style.display = "block"

    setInterval(() => {

        fetch(baseURl + "/status")
            .then(response => response.json())
            .then(data => {
                document.getElementById("cv-text").innerHTML = data["status"]
                console.log(data)
            })
            .catch(err => console.log(err))

        // Test API working in javascript
        // fetch("/status", {
        //     method: 'POST',
        //     mode: 'cors',
        //     cache: 'no-cache',
        //     credentials: 'same-origin',
        //     headers: {'Content-Type': 'application/json'},
        //     redirect: 'follow',
        //     referrerPolicy: 'no-referrer',
        //     body: JSON.stringify({})
        // }).then(response => response.json())
        //     .then(data => console.log(data))
        //     .catch(err => console.log(err))

    }, 1000)
}

if (img.complete) {
    loaded()
} else {
    img.addEventListener('load', loaded)
    img.addEventListener('error', function () {
        alert('error')
    })
}
