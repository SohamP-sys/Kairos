import csv,json,sys,unittest,tempfile,subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from assemble import assemble_prediction,sanitize_highlights
from schema import validate_output
from fpv_select import importance_curve,select_highlights
import numpy as np
class ProjectTests(unittest.TestCase):
 def test_assembly_valid(self):
  p=assemble_prediction("TEST","sample",30.0,raw_highlights=[{"segment":[0,8],"score":0.8},{"segment":[10,18],"score":0.5}])
  ok,err,warn=validate_output(p,expected_video_id="sample")
  self.assertTrue(ok,err); self.assertTrue(p["highlights"])
 def test_official_track4_label_rules_and_alias(self):
  skiing=assemble_prediction("TEST","skiing_03",30.0,label_name="skiing")
  self.assertEqual(skiing["label"]["name"],"ski")
  epic=assemble_prediction("TEST","P01_11",30.0,label_name="other")
  self.assertFalse(validate_output(epic,dataset="epic_kitchens_100")[0])
  fpv=assemble_prediction("TEST","Ski_video",30.0,label_name="cooking")
  self.assertFalse(validate_output(fpv,dataset="fpvsum")[0])
 def test_official_manifest_has_all_track4_videos(self):
  with (ROOT/"manifest.csv").open(newline="",encoding="utf-8") as f:
   rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),187)
  self.assertEqual(sum(r["dataset"]=="epic_kitchens_100" for r in rows),138)
  self.assertEqual(sum(r["dataset"]=="fpvsum" for r in rows),49)
 def test_fpv_downloader_extracts_youtube_id_before_segment_suffix(self):
  from tools.download_fpv import youtube_video_id
  self.assertEqual(youtube_video_id("BikePolo_5yf797Lh4Oc_1"),"5yf797Lh4Oc")
  self.assertEqual(youtube_video_id("Ski_-Z5a4UQmZLs"),"-Z5a4UQmZLs")
  with self.assertRaises(ValueError):
   youtube_video_id("Ski_short")
 def test_selector_budget_and_minimum(self):
  curve=np.linspace(0,1,90,dtype=np.float32)
  h=select_highlights(curve,90.0)
  self.assertTrue(h); self.assertTrue(all(e-s>=4.05 for s,e in [x["segment"] for x in h]))
  self.assertLessEqual(sum(e-s for s,e in [x["segment"] for x in h]),90*.15+0.1)
 def test_corrupt_video_fallback_writes_json(self):
  with tempfile.TemporaryDirectory() as d:
   v=Path(d)/"broken.mp4"; v.write_text("not video")
   out=Path(d)/"out.json"
   result=subprocess.run([sys.executable,str(ROOT/"code"/"run_video.py"),str(v),str(out)],check=False,timeout=60)
   self.assertNotEqual(result.returncode,0)
   self.assertTrue(out.exists())
   data=json.loads(out.read_text()); ok,err,_=validate_output(data,expected_video_id="broken")
   self.assertTrue(ok,err)
 def test_manifest_missing_video_placeholder(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); videos=d/"videos"; videos.mkdir(); outputs=d/"outputs"
   manifest=d/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nmissing_id,60,fpvsum\n",encoding="utf-8")
   subprocess.run([sys.executable,str(ROOT/"batch.py"),str(videos),str(outputs),"--manifest",str(manifest),"--fill-missing"],check=True,timeout=60)
   data=json.loads((outputs/"missing_id.json").read_text()); ok,err,_=validate_output(data,expected_video_id="missing_id",manifest_duration=60)
   self.assertTrue(ok,err)
   self.assertEqual(data["duration_sec"],60)
 def test_batch_fails_when_manifest_video_is_missing(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); videos=d/"videos"; videos.mkdir(); outputs=d/"outputs"
   manifest=d/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nmissing_id,60,fpvsum\n",encoding="utf-8")
   result=subprocess.run([sys.executable,str(ROOT/"batch.py"),str(videos),str(outputs),"--manifest",str(manifest)],check=False,timeout=60)
   self.assertNotEqual(result.returncode,0)
 def test_submission_validator_checks_manifest_coverage(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); outputs=d/"outputs"; outputs.mkdir()
   manifest=d/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nclip,30,fpvsum\n",encoding="utf-8")
   prediction=assemble_prediction("TEAM","clip",30,label_name="parkour",raw_highlights=[{"segment":[0,4.05]}])
   (outputs/"clip.json").write_text(json.dumps(prediction),encoding="utf-8")
   command=[sys.executable,str(ROOT/"validate.py"),"--dir",str(outputs),"--manifest",str(manifest)]
   self.assertEqual(subprocess.run(command,check=False,timeout=60).returncode,0)
   file_command=[sys.executable,str(ROOT/"validate.py"),"--file",str(outputs/"clip.json"),"--manifest",str(manifest),"-v"]
   self.assertEqual(subprocess.run(file_command,check=False,timeout=60).returncode,0)
   (outputs/"clip.json").unlink()
   self.assertNotEqual(subprocess.run(command,check=False,timeout=60).returncode,0)
 def test_stream_cleanup_only_removes_owned_video_directory(self):
  from tools.stream_manifest import OWNED_MARKER,_remove_download
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/"raw"; owned=root/"clip"; owned.mkdir(parents=True)
   (owned/OWNED_MARKER).write_text(json.dumps({"video_id":"clip","raw_root":str(root.resolve())}))
   (owned/"clip.mp4").write_bytes(b"video")
   _remove_download(owned,root)
   self.assertFalse(owned.exists())
   unowned=Path(d)/"unowned"; unowned.mkdir()
   with self.assertRaises(RuntimeError):
    _remove_download(unowned,root)
 def test_epic_download_runs_from_official_downloader_directory(self):
  from tools.stream_manifest import _download_video
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); downloader_dir=root/"official"; downloader_dir.mkdir()
   downloader=downloader_dir/"epic_downloader.py"; downloader.touch()
   raw=root/"raw"; video_work=raw/"P01_11"
   def download(command,check,cwd):
    self.assertTrue(check)
    self.assertEqual(cwd,downloader_dir)
    target=video_work/"P01_11.mp4"; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(b"video")
   with patch("tools.stream_manifest.subprocess.run",side_effect=download):
    result=_download_video("P01_11","epic_kitchens_100",video_work,downloader)
   self.assertEqual(result,video_work/"P01_11.mp4")
 def test_epic_download_prefers_md5_verifier_for_existing_video(self):
  from tools import stream_manifest
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); downloader_dir=root/"official"; downloader_dir.mkdir()
   downloader=downloader_dir/"epic_downloader.py"; downloader.touch()
   verifier=downloader_dir/"epic_video_downloader_verify.py"; verifier.touch()
   raw=root/"raw"; video_work=raw/"P01_11"
   target=video_work/"EPIC-KITCHENS"/"P01"/"videos"/"P01_11.MP4"
   target.parent.mkdir(parents=True)
   (video_work/stream_manifest.OWNED_MARKER).write_text(json.dumps({
    "video_id":"P01_11","raw_root":str(raw.resolve())
   }))
   target.write_bytes(b"possibly-truncated")
   def verify_download(command,check,cwd):
    self.assertTrue(check)
    self.assertEqual(cwd,downloader_dir)
    self.assertEqual(command[1],str(verifier))
    self.assertIn("--verify-ssl",command)
    target.write_bytes(b"md5-verified")
   with patch("tools.stream_manifest.subprocess.run",side_effect=verify_download) as run:
    result=stream_manifest._download_video("P01_11","epic_kitchens_100",video_work,downloader)
   self.assertEqual(result,target)
   run.assert_called_once()
 def test_stream_publishes_then_deletes_only_after_validation(self):
  from types import SimpleNamespace
  from tools import stream_manifest
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); manifest=root/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nclip,30,fpvsum\n")
   work=root/"work"; raw=work/"downloads"/"raw"; video_work=raw/"clip"; video_work.mkdir(parents=True)
   (video_work/stream_manifest.OWNED_MARKER).write_text(json.dumps({"video_id":"clip","raw_root":str(raw.resolve())}))
   source=video_work/"clip.mp4"; source.write_bytes(b"video")
   outputs=root/"outputs"
   def infer(command,check=False,env=None):
    candidate=Path(command[3])
    prediction=assemble_prediction("Kairos","clip",30,label_name="parkour",raw_highlights=[{"segment":[0,4.1]}])
    candidate.parent.mkdir(parents=True,exist_ok=True)
    candidate.write_text(json.dumps(prediction),encoding="utf-8")
    return SimpleNamespace(returncode=0)
   with patch.object(stream_manifest,"_require_models"), \
        patch.object(stream_manifest,"_download_video",return_value=source), \
        patch.object(stream_manifest.shutil,"disk_usage",return_value=SimpleNamespace(free=100*stream_manifest.GIB)), \
        patch.object(stream_manifest.subprocess,"run",side_effect=infer):
    result=stream_manifest.main([str(manifest),"--work-dir",str(work),"--output-dir",str(outputs),"--team-id","Kairos","--limit","1"])
   self.assertEqual(result,0)
   prediction=json.loads((outputs/"clip.json").read_text())
   self.assertTrue(validate_output(prediction,expected_video_id="clip",manifest_duration=30,dataset="fpvsum")[0])
   self.assertFalse(video_work.exists())
   self.assertTrue((work/"state"/"clip.done").is_file())
 def test_stream_prefetches_next_video_during_inference(self):
  from threading import Event
  from types import SimpleNamespace
  from tools import stream_manifest
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); manifest=root/"manifest.csv"
   manifest.write_text("video_id,duration_sec,dataset\nclip1,30,fpvsum\nclip2,30,fpvsum\n")
   work=root/"work"; raw=work/"downloads"/"raw"; outputs=root/"outputs"
   next_download_started=Event()
   def download(video_id,dataset,video_work,epic_downloader):
    video_work.mkdir(parents=True,exist_ok=True)
    (video_work/stream_manifest.OWNED_MARKER).write_text(json.dumps({
     "video_id":video_id,"raw_root":str(raw.resolve())
    }))
    source=video_work/(video_id+".mp4"); source.write_bytes(b"video")
    if video_id=="clip2":
     next_download_started.set()
    return source
   def infer(command,check=False,env=None):
    source=Path(command[2]); candidate=Path(command[3])
    if source.stem=="clip1":
     self.assertTrue(next_download_started.wait(timeout=2))
    candidate.parent.mkdir(parents=True,exist_ok=True)
    candidate.write_text(json.dumps(assemble_prediction(
     "Kairos",source.stem,30,label_name="parkour",
     raw_highlights=[{"segment":[0,4.1]}],
    )),encoding="utf-8")
    return SimpleNamespace(returncode=0)
   with patch.object(stream_manifest,"_require_models"), \
        patch.object(stream_manifest,"_download_video",side_effect=download), \
        patch.object(stream_manifest.shutil,"disk_usage",return_value=SimpleNamespace(free=100*stream_manifest.GIB)), \
        patch.object(stream_manifest.subprocess,"run",side_effect=infer):
    result=stream_manifest.main([
     str(manifest),"--work-dir",str(work),"--output-dir",str(outputs),
     "--team-id","Kairos","--dataset","fpvsum",
    ])
   self.assertEqual(result,0)
   for video_id in ("clip1","clip2"):
    self.assertTrue((outputs/(video_id+".json")).is_file())
    self.assertFalse((raw/video_id).exists())
 def test_stream_failure_retains_source_and_does_not_publish(self):
  from types import SimpleNamespace
  from tools import stream_manifest
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); manifest=root/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nclip,30,fpvsum\n")
   work=root/"work"; raw=work/"downloads"/"raw"; video_work=raw/"clip"; video_work.mkdir(parents=True)
   (video_work/stream_manifest.OWNED_MARKER).write_text(json.dumps({"video_id":"clip","raw_root":str(raw.resolve())}))
   source=video_work/"clip.mp4"; source.write_bytes(b"video")
   outputs=root/"outputs"
   def infer(command,check=False,env=None):
    candidate=Path(command[3]); candidate.write_text(json.dumps(assemble_prediction("wrong-team","clip",30,label_name="parkour",raw_highlights=[{"segment":[0,4.1]}])))
    return SimpleNamespace(returncode=0)
   with patch.object(stream_manifest,"_require_models"), \
        patch.object(stream_manifest,"_download_video",return_value=source), \
        patch.object(stream_manifest.shutil,"disk_usage",return_value=SimpleNamespace(free=100*stream_manifest.GIB)), \
        patch.object(stream_manifest.subprocess,"run",side_effect=infer):
    result=stream_manifest.main([str(manifest),"--work-dir",str(work),"--output-dir",str(outputs),"--team-id","Kairos","--limit","1"])
   self.assertEqual(result,1)
   self.assertTrue(source.is_file())
   self.assertFalse((outputs/"clip.json").exists())
 def test_avion_preparation_makes_flash_attention_optional(self):
  from tools.prepare_avion import prepare_avion_source
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/"transformer.py"
   source.write_text(
    "from flash_attn.modules.mha import MHA as FlashMHA\n"
    "from flash_attn.modules.mlp import Mlp as FlashMlp\n"
    "            self.attn = FlashMHA(d_model, n_head, cross_attn=False, bias=True, dropout=attn_drop, use_flash_attn=True)\n"
    "            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())\n",
    encoding="utf-8",
   )
   self.assertTrue(prepare_avion_source(source))
   prepared=source.read_text(encoding="utf-8")
   self.assertIn("if FlashMHA is None:",prepared)
   self.assertIn("if FlashMlp is None:",prepared)
   self.assertFalse(prepare_avion_source(source))
 def test_avion_extraction_progress_handles_multiple_batches(self):
  import tools.extract_avion_epic as extractor
  class FakeClassifier:
   enabled=True
   device="cpu"
   def __init__(self,*args): pass
   def load_action_mapping(self,path): return True
   def predict_joint_logits(self,clips,batch_size=1):
    return np.zeros((len(clips),3806),dtype=np.float32)
   def marginalize(self,logits):
    return np.full((len(logits),97),1/97,dtype=np.float32),np.full((len(logits),300),1/300,dtype=np.float32)
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); video=d/"source.mp4"; video.touch(); output=d/"cache.npz"
   frames=np.zeros((40,224,224,3),dtype=np.uint8)
   with patch.object(extractor,"probe_duration",return_value=5.0), \
        patch.object(extractor,"decode_frames",return_value=frames), \
        patch.object(extractor,"center_crop_224",return_value=frames), \
        patch.object(extractor,"AVIONEpicClassifier",FakeClassifier):
    extractor.extract_avion_cache(video,output,checkpoint_path=d/"checkpoint.pt",batch_size=1)
   self.assertTrue(output.is_file())
 def test_synthetic_video_pipeline(self):
  with tempfile.TemporaryDirectory() as d:
   v=Path(d)/"synthetic.mp4"; out=Path(d)/"out.json"
   subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","testsrc2=size=320x240:rate=15:duration=30","-f","lavfi","-i","sine=frequency=500:duration=30","-shortest","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(v)],check=True,timeout=60)
   subprocess.run([sys.executable,str(ROOT/"code"/"run_video.py"),str(v),str(out)],check=True,timeout=120)
   data=json.loads(out.read_text()); ok,err,_=validate_output(data,expected_video_id="synthetic")
   self.assertTrue(ok,err)
 def test_cooking_route_builds_missing_epic_cache(self):
  import pipeline
  import fpv_pipeline
  class CookingRouter:
   def route_and_label(self,frames,audio):
    return "other",0.9,{"other":0.9},np.zeros((0,64),dtype=np.float32)
  with tempfile.TemporaryDirectory() as d:
   video=Path(d)/"kitchen.mp4"
   video.touch()
   with patch.object(pipeline,"probe_duration",return_value=30.0), \
        patch.object(pipeline,"extract_motion_curve",return_value=[0.1]*30), \
        patch.object(pipeline,"decode_video_frames",return_value=np.zeros((0,224,224,3),dtype=np.uint8)), \
        patch.object(pipeline,"extract_audio_1hz_features",return_value=(np.zeros(30),np.zeros(30))), \
        patch.object(pipeline,"SigLIPRouter",return_value=CookingRouter()), \
        patch.object(pipeline,"_load_epic_cache",side_effect=[None,{"duration_sec":np.array([30.0])}]), \
        patch.object(pipeline,"_create_epic_cache") as create_cache, \
        patch("epic_actions.decode_epic_actions",return_value=[]) as decode, \
        patch("epic_select.select_epic_highlights",return_value=[{"segment":[0,6],"score":0.5}]), \
        patch.object(fpv_pipeline,"run_fpv_branch") as fpv:
    result=pipeline.process_video(video,dataset="epic_kitchens_100",manifest_duration=30.0)
   self.assertEqual(result["label"]["name"],"cooking")
   self.assertEqual(result["actions"],[])
   create_cache.assert_called_once_with(video,"kitchen")
   decode.assert_called_once()
   fpv.assert_not_called()
   self.assertEqual(result["duration_sec"],30.0)
   self.assertTrue(validate_output(result,expected_video_id="kitchen",manifest_duration=30.0,dataset="epic_kitchens_100")[0])
 def test_fpv_uses_importance_windows_when_kts_shots_exceed_budget(self):
  from fpv_pipeline import run_fpv_branch
  from videomae_fpv import FPV_CLASSES
  class Router:
   def category_prompt_curve(self,embeddings,label):
    return np.zeros(len(embeddings),dtype=np.float32)
  class VideoModel:
   enabled=False
   def predict(self,frames):
    return {name:0.0 for name in FPV_CLASSES}
  n=30
  with patch("builtins.print"):
   _,_,highlights,_,_=run_fpv_branch(
    Path("unused.mp4"),
    float(n),
    router=Router(),
    video_model=VideoModel(),
    frames=np.zeros((n,224,224,3),dtype=np.uint8),
    audio_flux=np.zeros(n,dtype=np.float32),
    highband=np.zeros(n,dtype=np.float32),
    motion=[0.0]*n,
    embeddings=np.ones((n,8),dtype=np.float32),
    sig_label="parkour",
    sig_score=0.8,
    sig_probs={"parkour":1.0},
   )
  self.assertTrue(highlights)
  total=sum(e-s for s,e in [x["segment"] for x in highlights])
  self.assertLessEqual(total,0.148*n+1e-3)
  self.assertTrue(all(e-s>=4.05 for s,e in [x["segment"] for x in highlights]))
if __name__=="__main__":unittest.main()


class TestKTSAndFPVModels(unittest.TestCase):
    def test_kts_segments_cover_timeline_and_minimum(self):
        import numpy as np
        from kts import kts_segments
        # Synthetic features with a distinct change every 8 seconds.
        x=np.vstack([np.tile([1.,0.],(8,1)),np.tile([0.,1.],(8,1)),np.tile([1.,0.],(8,1))])
        segs=kts_segments(x,min_len_sec=4.05,max_len_sec=12,penalty=0.05,fps=1.0)
        self.assertTrue(segs)
        self.assertGreaterEqual(segs[0][0],0)
        self.assertAlmostEqual(segs[-1][1],len(x),delta=0.1)
        self.assertTrue(all(e-s>=4.0 for s,e in segs))
    def test_videomae_label_alias_mapping(self):
        from videomae_fpv import LABEL_TO_CLASS, norm_label
        self.assertEqual(LABEL_TO_CLASS[norm_label("Mountain biking")], "bike")
        self.assertEqual(LABEL_TO_CLASS[norm_label("Rock climbing")], "rockclimbing")
        self.assertEqual(LABEL_TO_CLASS[norm_label("Scuba diving")], "scuba")
        self.assertNotIn(norm_label("playing basketball"), LABEL_TO_CLASS)

    def test_epic_official_vocabulary_and_action_mapping(self):
        from schema import load_supported_classes
        import numpy as np
        ROOT=Path(__file__).resolve().parents[1]
        vocab=load_supported_classes(ROOT/"supported_classes.json")
        self.assertEqual(len(vocab["verbs"]),98)
        self.assertEqual(len(vocab["nouns"]),301)
        with np.load(ROOT/"weights/epic_priors.npz",allow_pickle=False) as data:
            self.assertEqual(len(data["action_verb"]),3806)
            self.assertEqual(len(data["action_noun"]),3806)
            self.assertEqual(data["prior"].shape,(98,301))

    def test_video_mae_optional_without_checkpoint(self):
        from videomae_fpv import VideoMAEFpvClassifier, FPV_CLASSES
        model=VideoMAEFpvClassifier(model_path='__missing_checkpoint__')
        self.assertFalse(model.enabled)
        out=model.predict([])
        self.assertEqual(set(out),set(FPV_CLASSES))
        self.assertTrue(all(v==0 for v in out.values()))

    def test_kts_knapsack_respects_budget_and_no_overlap(self):
        from kts import knapsack_highlights
        segs=[[0,5],[6,11],[12,17],[18,23]]
        scores=[0.2,0.9,0.5,0.8]
        out=knapsack_highlights(segs,scores,100,budget_ratio=0.15,min_len=4.05)
        chosen=[x["segment"] for x in out]
        self.assertLessEqual(sum(e-s for s,e in chosen),15.0+1e-3)
        self.assertTrue(all(e-s>=4.0 for s,e in chosen))
        self.assertTrue(all(chosen[i][1]<=chosen[i+1][0] for i in range(len(chosen)-1)))
