"""Phase 4 HTTP tests (FastAPI TestClient, SYNTHETIC model bundle, no client data).   python -m unittest tests.test_phase4_api -v
Needs the web dependencies:  pip install -r requirements-dev.txt   (skipped automatically if fastapi/httpx are missing)."""
import sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import joblib, pandas as pd
from kestrel import final as FIN
from tests.synthetic_bundle import make_bundle, valid_claim, BAD_PARTNER

try:
    from fastapi.testclient import TestClient
    from app import create_app
    HAVE_WEB = True
except Exception:                      # fastapi / httpx not installed
    HAVE_WEB = False


@unittest.skipUnless(HAVE_WEB, "fastapi/httpx not installed: pip install -r requirements-dev.txt")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.bundle = make_bundle(); path = Path(cls.tmp.name) / "bundle.joblib"
        joblib.dump(cls.bundle, path)
        cls.cm = TestClient(create_app(path)); cls.c = cls.cm.__enter__()          # runs startup: model loaded once

    @classmethod
    def tearDownClass(cls): cls.cm.__exit__(None, None, None); cls.tmp.cleanup()

    def post(self, body=None, **kw): return self.c.post("/predict", json=body if body is not None else valid_claim(), **kw)

    def test_health(self):
        r = self.c.get("/health"); self.assertEqual(r.status_code, 200); self.assertEqual(r.json(), {"status": "ok", "model_loaded": True})

    def test_valid_predict(self):
        r = self.post(); self.assertEqual(r.status_code, 200); d = r.json()
        self.assertIsInstance(d["fraud_score"], float); self.assertTrue(0 <= d["fraud_score"] <= 1)
        self.assertIn(d["risk_level"], ("low", "medium", "high")); self.assertTrue(1 <= len(d["reasons"]) <= 3)
        self.assertIn("not automatic rejection", d["note"])

    def test_api_agrees_with_phase3_scoring_function(self):
        for over in ({}, {"partner_id": BAD_PARTNER}):
            claim = valid_claim(**over); d = self.post(claim).json()
            raw = pd.DataFrame([{**claim, "product_serial": "", "claim_description": "", "inspector_note": ""}])
            _, p = FIN.score(raw, self.bundle); self.assertAlmostEqual(d["fraud_score"], float(p[0]), places=7)

    def test_missing_required_field(self):
        c = valid_claim(); c.pop("customer_prior_claims"); r = self.post(c)
        self.assertEqual(r.status_code, 422); e = r.json()["errors"]; self.assertEqual(e[0]["field"], "customer_prior_claims")
        self.assertNotIn("Traceback", r.text)

    def test_malformed_values(self):
        for over, field in (({"claim_amount_inr": "abc"}, "claim_amount_inr"), ({"submitted_at": "not a date"}, "submitted_at"),
                            ({"photo_attached": "maybe"}, "photo_attached"), ({"sku": "NOPE"}, "sku"), ({"days_since_purchase": -4}, "days_since_purchase")):
            r = self.post(valid_claim(**over)); self.assertEqual(r.status_code, 422, over); self.assertEqual(r.json()["errors"][0]["field"], field)

    def test_malformed_json_and_wrong_shapes(self):
        r = self.c.post("/predict", content="{not json", headers={"Content-Type": "application/json"}); self.assertEqual(r.status_code, 400)
        self.assertEqual(self.c.post("/predict", content=b"\xff\xfe", headers={"Content-Type": "application/json"}).status_code, 400)
        self.assertEqual(self.c.post("/predict", content="", headers={"Content-Type": "application/json"}).status_code, 400)
        for body in ([], "text", 5, None): self.assertEqual(self.post(body if body is not None else "null").status_code, 422)
        self.assertEqual(self.c.post("/predict", content="x" * 70_000).status_code, 413)

    def test_unseen_partner_does_not_crash(self):
        r = self.post(valid_claim(partner_id="SP9999")); self.assertEqual(r.status_code, 200); d = r.json()
        self.assertFalse(d["partner_known"]); self.assertTrue(0 <= d["fraud_score"] <= 1)

    def test_missing_optional_note(self):
        c = valid_claim(); c.pop("inspector_note"); self.assertEqual(self.post(c).status_code, 200)

    def test_service_survives_errors(self):
        self.post({}); self.post("[]"); self.assertEqual(self.post().status_code, 200)

    def test_wrong_method_and_unknown_path(self):
        self.assertEqual(self.c.get("/predict").status_code, 405); self.assertEqual(self.c.get("/nope").status_code, 404)

    def test_ui_served_and_meta(self):
        r = self.c.get("/"); self.assertEqual(r.status_code, 200); self.assertIn("Check Claim", r.text)
        self.assertIn("A high score is not a fraud verdict", r.text)
        m = self.c.get("/meta").json(); self.assertIn("skus", m); self.assertEqual(m["risk_thresholds"]["high"], 0.13)

    def test_missing_model_is_reported_not_crashed(self):
        with TestClient(create_app(Path(self.tmp.name) / "does_not_exist.joblib")) as c:
            h = c.get("/health"); self.assertEqual(h.status_code, 200); self.assertFalse(h.json()["model_loaded"])
            self.assertEqual(c.post("/predict", json=valid_claim()).status_code, 503)


if __name__ == "__main__":
    unittest.main()
