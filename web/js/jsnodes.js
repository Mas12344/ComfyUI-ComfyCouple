import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "loaders.jsnodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("loaders")) {
			return;
		  }
		switch (nodeData.name) {

			case "ComfyCoupleMask":

				nodeType.prototype.onNodeCreated = function () {
				this._type = "COUPLE_REGION"
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.addWidget("button", "Update inputs", null, () => {
					this.inputs = this.inputs.slice(0,this.inputs_offset+2)

					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
					for(let i = 0; i < target_number_of_inputs; ++i)
						this.addInput(`region_${i+1}`, this._type)

					});
				}
				break;
						
		}	
		
	}
});