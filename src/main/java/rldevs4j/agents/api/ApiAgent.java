package rldevs4j.agents.api;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import rldevs4j.base.agent.Agent;
import rldevs4j.base.agent.preproc.Preprocessing;
import rldevs4j.base.env.msg.Event;
import rldevs4j.base.env.msg.EventType;
import rldevs4j.base.env.msg.Step;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class ApiAgent extends Agent {
    private double cumReward;
    private String urlPath;
    private Map<String, Object> params;
    private HttpURLConnection con;
    private ObjectMapper jsonMapper;

    public ApiAgent(String name, Preprocessing preprocessing, String url, Map<String, Object> params) throws MalformedURLException {
        super(name, preprocessing, 0);
        this.urlPath = url;
        this.params = params;
        this.jsonMapper = new ObjectMapper();
        this.jsonMapper.configure(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES, true);
        this.jsonMapper.setAnnotationIntrospector(new Step.IgnoreInheritedIntrospector());
        this.initAgent(params);
    }

    @Override
    public Event observation(Step step) {
        //add step reward
        this.cumReward += step.getReward();
        // Send step to Agent API
        try {
            URL url = new URL(urlPath+"step");
            this.con = (HttpURLConnection) url.openConnection();
            // Set “content-type” request header to "application/json" to send the request content in JSON form.
            this.con.setRequestProperty("Content-Type", "application/json; utf-8");
            // Set the “Accept” request header to "application/json" to read the response in json format
            this.con.setRequestProperty("Accept", "application/json");
            // To send request content
            this.con.setDoOutput(true);
            // sent request
            String response = sentRequest("POST", object2Json(step));
            TypeReference<HashMap<String,Integer>> typeRef = new TypeReference<HashMap<String,Integer>>() {};
            Map<String, Integer> resp = jsonMapper.readValue(response, typeRef);
            Event action = new Event(resp.get("action"),"action", EventType.action);
            return action;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public double getTotalReward() {
        return cumReward;
    }

    @Override
    public void trainingFinished() {

    }

    @Override
    public void clear() {
        cumReward = 0D;
    }

    @Override
    public void saveModel(String path) {

    }

    @Override
    public void loadModel(String path) throws IOException {

    }

    private void initAgent(Map<String, Object> params){
        try {
            URL url = new URL(urlPath+"init");
            this.con = (HttpURLConnection) url.openConnection();
            // Set “content-type” request header to "application/json" to send the request content in JSON form.
            this.con.setRequestProperty("Content-Type", "application/json; utf-8");
            // Set the “Accept” request header to "application/json" to read the response in json format
            this.con.setRequestProperty("Accept", "application/json");
            // To send request content
            this.con.setDoOutput(true);
            // sent request
            String response = sentRequest("POST", object2Json(params));
            TypeReference<HashMap<String,Object>> typeRef = new TypeReference<HashMap<String,Object>>() {};
            Map<String, Object> resp = jsonMapper.readValue(response, typeRef);
            System.out.println(resp.get("status").equals("OK")?"Agent Initialized.":"Agent Initialization Error.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String object2Json(Object o) throws JsonProcessingException {
        String json = jsonMapper.writeValueAsString(o);
        return json;
    }

    private String sentRequest(String type, String content){
        StringBuilder response = new StringBuilder();
        try {
            con.setRequestMethod(type); // POST/GET/PUT
            try(OutputStream os = con.getOutputStream()) {
                byte[] input = content.getBytes("utf-8");
                os.write(input, 0, input.length);
            }
            try(BufferedReader br = new BufferedReader(new InputStreamReader(con.getInputStream(), "utf-8"))) {
                String responseLine = null;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
//                System.out.println(response.toString());
            }
        } catch (ProtocolException | UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return response.toString();
    }

}
