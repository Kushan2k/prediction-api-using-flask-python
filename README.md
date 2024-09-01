git clone https://github.com/your-username/solo-traveller-api.git

````

2. **Navigate to the project directory:**
```bash
cd solo-traveller-api
````

3. **Create a virtual environment:**

   ```bash
   python3 -m venv env
   ```

4. **Activate the virtual environment:**

   ```bash
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate  # Windows
   ```

5. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

6. **Set up environment variables:**

   - Create a `.env` file in the project root directory.
   - Add the following environment variables:

     ```
      CONNECTION_STRING=
      PASSWORD=
      USERNAME=
      TF_ENABLE_ONEDNN_OPTS=
      PORT=
      DEBUG=

     ```

7. **Run the application:**
   ```bash
   python app.py
   ```

## Usage

The API provides the following endpoints:

- **`/`:** save user informations.
- **`/upload`:** image classification for upload image and get the recommneted categories and stores.
- **`/predict`:** predict the user category and return the predicted stores and recomendations.

## Contributing

Contributions are welcome! Please submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
