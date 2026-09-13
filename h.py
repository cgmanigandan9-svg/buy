import re
import sys
from pathlib import Path
import pandas as pd

# Optional OCR integration (e.g., pytesseract / EasyOCR / LLM visual extraction)
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class FinancialAgentPipeline:
    def __init__(self, data_dir: str = "dataset"):
        configured_dir = Path(data_dir)
        if configured_dir.is_absolute():
            self.data_dir = configured_dir
        else:
            script_dir = Path(__file__).resolve().parent
            candidates = [Path.cwd() / configured_dir, script_dir / configured_dir, script_dir.parent / configured_dir]
            self.data_dir = next((path for path in candidates if path.is_dir()), candidates[0])
        self.load_datasets()

    def load_datasets(self):
        """Loads all CSV datasets into pandas DataFrames."""
        required_columns = {
            "requests.csv": {"request_id", "user_id", "request_date", "requested_amount"},
            "financial_profiles.csv": {"user_id", "available_balance", "minimum_balance"},
            "financial_events.csv": {"event_id", "amount"},
            "exchange_rates.csv": {"from_currency", "to_currency", "rate_date", "rate"},
            "request_payment_options.csv": {"request_id", "payment_option_id"},
            "messages.csv": set(),
        }
        missing_files = [name for name in required_columns if not (self.data_dir / name).is_file()]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Missing dataset file(s): {missing}. Expected them in '{self.data_dir}'."
            )

        datasets = {}
        for filename, columns in required_columns.items():
            frame = pd.read_csv(self.data_dir / filename)
            missing_columns = sorted(columns - set(frame.columns))
            if missing_columns:
                raise ValueError(
                    f"'{filename}' is missing required column(s): {', '.join(missing_columns)}."
                )
            datasets[filename] = frame

        self.requests = datasets["requests.csv"]
        self.profiles = datasets["financial_profiles.csv"]
        self.events = datasets["financial_events.csv"]
        self.exchange_rates = datasets["exchange_rates.csv"]
        self.payment_options = datasets["request_payment_options.csv"]
        self.messages = datasets["messages.csv"]

        for frame, columns in [
            (self.requests, ["requested_amount"]),
            (self.profiles, ["available_balance", "minimum_balance"]),
            (self.events, ["amount"]),
            (self.exchange_rates, ["rate"]),
        ]:
            for column in columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if self.requests.empty:
            raise ValueError("'requests.csv' does not contain any requests to evaluate.")

        for filename, frame, columns in [
            ("requests.csv", self.requests, ["requested_amount"]),
            ("financial_profiles.csv", self.profiles, ["available_balance", "minimum_balance"]),
            ("exchange_rates.csv", self.exchange_rates, ["rate"]),
        ]:
            invalid_rows = frame[columns].isna().any(axis=1)
            if invalid_rows.any():
                rows = ", ".join(str(index + 2) for index in frame.index[invalid_rows][:5])
                raise ValueError(
                    f"'{filename}' contains invalid numeric value(s) near CSV row(s): {rows}."
                )

        images_path = self.data_dir / "images.csv"
        self.images = pd.read_csv(images_path) if images_path.is_file() else pd.DataFrame()

    def extract_ocr_amount(self, image_id: str) -> float:
        """Extracts numeric financial amount from image payload if OCR is present."""
        if not HAS_OCR or self.images.empty:
            return 0.0
        
        image_path = self.data_dir / "media" / "images" / f"{image_id}.png"
        if not image_path.is_file():
            return 0.0
        
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            # Basic regex pattern extraction for currency amounts
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            return float(numbers[0]) if numbers else 0.0
        except Exception:
            return 0.0

    def convert_currency(self, amount: float, from_curr: str, to_curr: str, date_str: str) -> float:
        """Converts currency amount to user home currency using fixed FX table."""
        if from_curr == to_curr or pd.isna(from_curr) or pd.isna(amount) or amount == 0:
            return amount
        
        fx_match = self.exchange_rates[
            (self.exchange_rates["from_currency"] == from_curr) & 
            (self.exchange_rates["to_currency"] == to_curr) &
            (self.exchange_rates["rate_date"] == date_str)
        ]
        
        if not fx_match.empty:
            return float(amount) * float(fx_match.iloc[0]["rate"])
        
        # Fallback to general currency rate without date restriction if exact date is missing
        fx_fallback = self.exchange_rates[
            (self.exchange_rates["from_currency"] == from_curr) & 
            (self.exchange_rates["to_currency"] == to_curr)
        ]
        return float(amount) * float(fx_fallback.iloc[0]["rate"]) if not fx_fallback.empty else amount

    def preprocess_events(self):
        """Imputes missing amounts in events using image OCR data."""
        for idx, row in self.events.iterrows():
            if pd.isna(row["amount"]) or row["amount"] == 0:
                event_id = row["event_id"]
                if not self.images.empty and "related_event_id" in self.images.columns:
                    img_match = self.images[self.images["related_event_id"] == event_id]
                    if not img_match.empty:
                        image_id = img_match.iloc[0]["image_id"]
                        self.events.at[idx, "amount"] = self.extract_ocr_amount(image_id)

    def calculate_safe_amount(self, user_profile: pd.Series, request_date: str) -> float:
        """Calculates current liquid margin above required minimum balance."""
        available_balance = user_profile["available_balance"]
        min_balance = user_profile["minimum_balance"]
        if pd.isna(available_balance) or pd.isna(min_balance):
            return 0.0
        safe_margin = max(0.0, available_balance - min_balance)
        return safe_margin

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        missing = pd.isna(value)
        return False if isinstance(missing, bool) and missing else bool(value)

    def evaluate_request(self, req: pd.Series) -> dict:
        """Evaluates financial requests and formats decision attributes."""
        user_id = req["user_id"]
        req_id = req["request_id"]
        req_date = req["request_date"]
        requested_amt = pd.to_numeric(req["requested_amount"], errors="coerce")
        allows_partial = self._as_bool(req.get("allows_partial_payment", False))

        if pd.isna(requested_amt) or requested_amt < 0:
            raise ValueError(f"Request '{req_id}' has an invalid requested_amount: {requested_amt!r}.")

        matching_profiles = self.profiles[self.profiles["user_id"] == user_id]
        if matching_profiles.empty:
            raise ValueError(f"No financial profile found for user '{user_id}'.")
        user_prof = matching_profiles.iloc[0]
        safe_to_pay_limit = self.calculate_safe_amount(user_prof, req_date)
        
        amount_safe_to_pay = min(safe_to_pay_limit, requested_amt)
        
        # Decision Logic
        if amount_safe_to_pay >= requested_amt:
            status = "affordable_now"
            rec_method = "full_payment"
            earliest_date = req_date
            spending_changes = ""
            plan = f"{req_date}:{requested_amt:.2f}"
            explanation = f"User has sufficient buffer above required minimum balance ({user_prof['minimum_balance']})."
        elif amount_safe_to_pay > 0 and allows_partial:
            status = "affordable_with_plan"
            rec_method = "partial_payment"
            earliest_date = ""
            spending_changes = "reduce_discretionary_spending"
            plan = f"{req_date}:{amount_safe_to_pay:.2f}"
            explanation = f"Partial payment safe up to available liquid margin ({amount_safe_to_pay:.2f})."
        else:
            # Check options for installments or future income
            user_options = self.payment_options[self.payment_options["request_id"] == req_id]
            if not user_options.empty:
                status = "affordable_with_plan"
                rec_method = "installments"
                earliest_date = ""
                spending_changes = ""
                plan = f"installment_plan_option_{user_options.iloc[0]['payment_option_id']}"
                explanation = "Request affordable using structured installment payment plan."
            else:
                status = "not_affordable"
                rec_method = "none"
                earliest_date = ""
                spending_changes = "stop_non_essential_spending"
                plan = ""
                explanation = "Insufficient balance and liquid forecast to safely support requested payment."

        return {
            "request_id": req_id,
            "amount_safe_to_pay": round(amount_safe_to_pay, 2),
            "affordability_status": status,
            "recommended_payment_method": rec_method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": earliest_date,
            "spending_changes_needed": spending_changes,
            "decision_explanation": explanation
        }

    def run(self, output_path: str = "output.csv"):
        """Executes the evaluation pipeline and writes the results to CSV."""
        self.preprocess_events()
        results = []
        for _, req in self.requests.iterrows():
            results.append(self.evaluate_request(req))
        
        out_df = pd.DataFrame(results)
        columns_order = [
            "request_id",
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
            "decision_explanation"
        ]
        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = Path(__file__).resolve().parent.parent / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_df[columns_order].to_csv(output_file, index=False)
        print(f"Pipeline executed successfully. Output saved to {output_path}")


if __name__ == "__main__":
    try:
        pipeline = FinancialAgentPipeline(data_dir="dataset")
        pipeline.run()
    except (FileNotFoundError, ValueError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        sys.exit(1)