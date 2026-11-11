"""Local mock-process tests for the industrial GUI, NOT scientific validation."""
import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "studio"))
import server as s

def fake_structure(path):
    if not Path(path).read_bytes():
        raise ValueError("empty")
    return {"formula":"In4Sb4","atom_count":8,
        "cell":[[6.5,0,0],[0,6.5,0],[0,0,6.5]],
        "pbc":[True,True,True],
        "symbols":["In"]*4+["Sb"]*4,
        "positions":[[i%2,(i//2)%2,i//4] for i in range(8)],
        "preview_truncated":False,"lengths":[6.5]*3,"angles":[90]*3}

class TestIndustrialGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        s.DATA=cls.root/"workspace"
        s.SOURCE=cls.root/"source"
        (s.SOURCE/"examples").mkdir(parents=True)
        (s.SOURCE/"examples"/"InSb.cif").write_text("mock crystal sample")
        s.BGP_BIN=Path(__file__).resolve().with_name("fake_bgp.sh")
        s.BGP_BIN.chmod(0o755)
        s.DEFAULT_ABACUS = cls.root / "abacus-install" / "bin" / "abacus_pw_gpu"
        s.DEFAULT_ABACUS.parent.mkdir(parents=True)
        s.DEFAULT_ABACUS.write_text("#!/bin/sh\nexit 0\n")
        s.DEFAULT_ABACUS.chmod(0o755)
        s.BGP_DEPS_PREFIX = cls.root / "bgp-deps"
        (s.BGP_DEPS_PREFIX / "lib").mkdir(parents=True)
        cls.asepatch=patch.object(s,"parse_crystal",fake_structure)
        cls.asepatch.start()
        s.init_store()
        threading.Thread(target=s.worker,daemon=True).start()
        cls.httpd=s.ThreadingHTTPServer(("127.0.0.1",0),s.App)
        cls.port=cls.httpd.server_address[1]
        cls.serverthread=threading.Thread(target=cls.httpd.serve_forever,daemon=True)
        cls.serverthread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.asepatch.stop()
        s.db().close()
        cls.tmp.cleanup()

    def req(self,method,path,data=None,headers=None):
        h={"X-BGP-UI":"desktop"}
        if data is not None: h["Content-Type"]="application/json"
        h.update(headers or {})
        cn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=15)
        cn.request(method,path,body=data,headers=h)
        res=cn.getresponse()
        raw=res.read()
        code=res.status
        cn.close()
        try: return code,json.loads(raw)
        except (ValueError,UnicodeError): return code,raw

    def post(self,path,body):
        return self.req("POST",path,json.dumps(body).encode())

    def test_a_system_and_files(self):
        code,data=self.req("GET","/api/system")
        self.assertEqual(code,200)
        self.assertTrue(data["installed"])
        self.assertEqual(len(data["methods"]),4)
        for url in ("/","/static/style.css","/static/app.js"):
            c,raw=self.req("GET",url)
            self.assertEqual(c,200)
            self.assertGreater(len(raw),1000)

    def test_b_job_lifecycle(self):
        c,samples=self.req("GET","/api/samples")
        self.assertEqual(c,200)
        self.assertEqual(samples[0]["filename"],"InSb.cif")
        c,m=self.post("/api/materials/from-sample",{"filename":"InSb.cif"})
        self.assertEqual(c,201)
        self.assertEqual(m["formula"],"In4Sb4")
        c,j=self.post("/api/jobs",{"material_id":m["id"],"backend":"gnn-hse06","device":"cpu"})
        self.assertEqual(c,201)
        limit=time.monotonic()+10
        while time.monotonic()<limit:
            _,job=self.req("GET","/api/jobs/"+j["id"])
            if job["state"] in s.TERMINAL: break
            time.sleep(.08)
        self.assertEqual(job["state"],"success",job)
        self.assertEqual(job["result"]["prediction"]["gap_ev"],0.2487309)
        self.assertEqual(job["result"]["prediction"]["gap_type"],"direct")
        self.assertEqual(job["result"]["method"],"SIMULATED TEST ONLY")
        c,logs=self.req("GET","/api/jobs/"+j["id"]+"/log")
        self.assertEqual(c,200)
        self.assertIn("Loading verified",logs["text"])
        c,files=self.req("GET","/api/jobs/"+j["id"]+"/files")
        self.assertEqual(c,200)
        self.assertTrue(any(f["path"].endswith("result.json") for f in files))
        rel=next(f["path"] for f in files if f["path"].endswith("result.json"))
        c,result=self.req("GET","/api/jobs/"+j["id"]+"/download?path="+rel)
        self.assertEqual(c,200)
        self.assertEqual(result["run_id"],"testing-0001")
        c,bad=self.req("GET","/api/jobs/"+j["id"]+"/download?path=../../../../etc/passwd")
        self.assertEqual(c,400)
        c,copy=self.post("/api/jobs/"+j["id"]+"/duplicate",{})
        self.assertEqual(c,201)
        self.assertNotEqual(copy["id"],j["id"])
        c,removed=self.post("/api/materials/"+m["id"]+"/remove",{})
        self.assertEqual(c,200)
        self.assertTrue(removed["ok"])
        self.assertTrue(removed["archived"])
        self.assertGreaterEqual(removed["jobs_preserved"],1)
        _,active=self.req("GET","/api/materials")
        self.assertNotIn(m["id"],[x["id"] for x in active])
        # A removed material cannot be submitted as a NEW active material.
        c,bad=self.post("/api/jobs",{"material_id":m["id"],"backend":"gnn-hse06","device":"cpu"})
        self.assertEqual(c,400)
        # Existing job and its frozen input remain reproducible after removal.
        c,dup=self.post("/api/jobs/"+j["id"]+"/duplicate",{})
        self.assertEqual(c,201)

    def test_c_diagnostics(self):
        c,response=self.post("/api/diagnostic",{"action":"assets"})
        self.assertEqual(c,200)
        self.assertTrue(response["ok"])
        c,response=self.post("/api/diagnostic",{"action":"doctor","abacus":""})
        self.assertEqual(c,200)
        self.assertFalse(response["ok"])

    def test_e_real_backend_routing_mock(self):
        # Validate the four actual CLI backend identifiers, not scientific predictions.
        c,m=self.post("/api/materials/from-sample",{"filename":"InSb.cif"})
        self.assertEqual(c,201)
        for backend in ("gnn-hse06","abacus-hse06","abacus-pbe0","cider-pbe0"):
            with self.subTest(backend=backend):
                payload={"material_id":m["id"],"backend":backend,"device":"cpu"}
                if backend!="gnn-hse06":
                    payload["abacus"]=str(s.DEFAULT_ABACUS)
                code,job=self.post("/api/jobs",payload)
                self.assertEqual(code,201)
                deadline=time.monotonic()+9
                while time.monotonic()<deadline:
                    _,job=self.req("GET","/api/jobs/"+job["id"])
                    if job["state"] in s.TERMINAL:
                        break
                    time.sleep(.07)
                self.assertEqual(job["state"],"success",job)
                self.assertEqual(job["backend"],backend)
                self.assertEqual(job["result"]["backend"],backend)
                self.assertEqual(job["result"]["method"],"SIMULATED TEST ONLY")
                _,logs=self.req("GET",f'/api/jobs/{job["id"]}/log')
                self.assertIn("MOCK TEST BACKEND: "+backend,logs["text"])
                if backend!="gnn-hse06":
                    self.assertIn("MOCK ABACUS PATH: "+str(s.DEFAULT_ABACUS),logs["text"])
                    self.assertIn("[GUI] Work root:",logs["text"])
                    self.assertIn("[GUI] ABACUS binary:",logs["text"])

    def test_f_automatic_abacus_configuration(self):
        code,system=self.req("GET","/api/system")
        self.assertEqual(code,200)
        self.assertEqual(system["abacus"]["path"],str(s.DEFAULT_ABACUS))
        self.assertTrue(system["abacus"]["executable"])
        self.assertTrue(system["abacus"]["deps_lib_present"])
        c,m=self.post("/api/materials/from-sample",{"filename":"InSb.cif"})
        self.assertEqual(c,201)
        # GPU methods can use the launcher-configured binary without typing a path.
        for method in ("abacus-hse06","cider-pbe0"):
            c,j=self.post("/api/jobs",{"material_id":m["id"],"backend":method,"device":"cpu"})
            self.assertEqual(c,201,j)
            self.assertEqual(j["abacus_path"],str(s.DEFAULT_ABACUS))
            self.assertNotIn("/mnt/d/",j["run_dir"] or "")
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                _,done=self.req("GET","/api/jobs/"+j["id"])
                if done["state"] in s.TERMINAL: break
                time.sleep(.08)
            self.assertEqual(done["state"],"success",done)
            c,logs=self.req("GET","/api/jobs/"+j["id"]+"/log")
            self.assertIn("MOCK ABACUS PATH: "+str(s.DEFAULT_ABACUS),logs["text"])
        # Reject invalid GPU executable before queuing or starting a fake calculation.
        c,error=self.post("/api/jobs",{"material_id":m["id"],"backend":"abacus-hse06",
                                       "abacus":str(s.DEFAULT_ABACUS)+".missing"})
        self.assertEqual(c,400)
        self.assertIn("不可执行",error["error"])

    def test_g_xyz_upload_then_three_backends(self):
        # UI-imported XYZ follows the real request/queue/job/result protocol.
        # Scientific predictions remain MOCK ONLY in this isolated test.
        xyz=b"8\\nmock periodic InSb input (ASE parser mocked)\\n"
        code,m=self.req("POST","/api/materials/upload",xyz,
                        {"X-Filename":"InSb.xyz","Content-Type":"application/octet-stream"})
        self.assertEqual(code,201,m)
        self.assertEqual(m["filename"],"InSb.xyz")
        for backend in ("gnn-hse06","abacus-hse06","cider-pbe0"):
            c,j=self.post("/api/jobs",{"material_id":m["id"],"backend":backend,"device":"cpu"})
            self.assertEqual(c,201,j)
            self.assertEqual(j["backend"],backend)
            if backend!="gnn-hse06":
                self.assertEqual(j["abacus_path"],str(s.DEFAULT_ABACUS))
            else:
                self.assertEqual(j["abacus_path"],"")
            deadline=time.monotonic()+9
            while time.monotonic()<deadline:
                _,done=self.req("GET","/api/jobs/"+j["id"])
                if done["state"] in s.TERMINAL: break
                time.sleep(.07)
            self.assertEqual(done["state"],"success",done)
            self.assertIsNotNone(done["result"])
            c,files=self.req("GET","/api/jobs/"+j["id"]+"/files")
            self.assertEqual(c,200)
            self.assertTrue(any(x["path"].endswith("/InSb.xyz") for x in files),files)

    def test_d_security_and_management(self):
        c,response=self.req("POST","/api/jobs",b"{}",{"X-BGP-UI":"wrong"})
        self.assertEqual(c,403)
        self.assertEqual(s.clean_basename(r"C:\folder\Si.cif"),"Si.cif")
        with self.assertRaises(ValueError):
            s.clean_basename("bad.exe")
        c,m=self.post("/api/materials/from-sample",{"filename":"InSb.cif"})
        self.assertEqual(c,201)
        c,new=self.post("/api/materials/"+m["id"]+"/rename",{"label":"测试 InSb"})
        self.assertEqual(c,200)
        self.assertEqual(new["label"],"测试 InSb")
        c,done=self.post("/api/materials/"+m["id"]+"/remove",{})
        self.assertEqual(c,200)
        self.assertTrue(done["ok"])

if __name__=="__main__":
    unittest.main()
