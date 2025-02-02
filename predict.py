from setfit import SetFitModel
saved_model = SetFitModel._from_pretrained(save_directory)
test_texts = ["Book me a hotel", "need a room with wifi","which model are you?", "I want to learn python","source is indore, destination is dehradun","your site is bad","accept paypal?","Discount?","Can somebody please help?","payment is stuck"]
preds_intent, preds_domain, preds_hitl = model(test_texts)

print("Intent Predictions:", [intent_map[p.item()] for p in preds_intent])
print("Domain Predictions:", [domain_map[p.item()] for p in preds_domain])
print("HITL Predictions:", [hitl_map[p.item()] for p in preds_hitl])
