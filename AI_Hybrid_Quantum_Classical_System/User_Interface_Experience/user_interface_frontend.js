
// /User_Interface_Experience/user_interface_frontend.js

class UI {
    constructor() {
        this.initListeners();
    }

    initListeners() {
        document.getElementById('submitBtn').addEventListener('click', () => {
            this.handleSubmit();
        });
    }

    handleSubmit() {
        const data = document.getElementById('inputField').value;
        this.sendDataToServer(data);
    }

    sendDataToServer(data) {
        fetch('/api/data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: data }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            this.updateUI(data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }

    updateUI(data) {
        document.getElementById('responseContainer').innerText = data.message;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new UI();
});
