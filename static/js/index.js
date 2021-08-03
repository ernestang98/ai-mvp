// Not sure if lines 3 - 6 is needed for production

let baseURl = "SET_BASE_URL_HERE"

if (location.hostname === "localhost" || location.hostname === "0.0.0.0" || location.hostname === "127.0.0.1")
    baseURl = ""

setInterval(() => {

    fetch(baseURl + "/status")
        .then(response => response.json())
        .then(data => {
            document.getElementById("test").innerHTML = data["status"]
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

}, 100)